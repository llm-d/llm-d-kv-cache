/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package engineadapter

import (
	"context"
	"encoding/binary"
	"fmt"
	"strings"
	"time"

	zmq "github.com/pebbe/zmq4"
	"github.com/vmihailenco/msgpack/v5"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents/events"
)

const (
	// vLLM event type tags.
	eventTagBlockStored      = "BlockStored"
	eventTagBlockRemoved     = "BlockRemoved"
	eventTagAllBlocksCleared = "AllBlocksCleared"

	defaultDeviceTier = "gpu"
	// pollTimeout is how often the poller should time out to check for context cancellation.
	pollTimeout = 250 * time.Millisecond
)

// VLLMAdapter implements the EngineAdapter interface for vLLM engines.
// It directly owns the ZMQ socket and handles msgpack encoding/decoding.
type VLLMAdapter struct {
	socket          *zmq.Socket
	poller          *zmq.Poller
	eventConverters map[string]func([]byte) (events.GenericEvent, error)
}

// NewVLLMAdapter creates a new vLLM adapter.
// The ZMQ socket is created lazily on the first Connect or Bind call.
func NewVLLMAdapter() (*VLLMAdapter, error) {
	adapter := &VLLMAdapter{}

	// Initialize event converters map
	adapter.eventConverters = map[string]func([]byte) (events.GenericEvent, error){
		eventTagBlockStored:      adapter.convertBlockStoredEvent,
		eventTagBlockRemoved:     adapter.convertBlockRemovedEvent,
		eventTagAllBlocksCleared: adapter.convertAllBlocksClearedEvent,
	}

	return adapter, nil
}

// ensureSocket creates a fresh SUB socket only if the current one is nil.
// If the socket is still valid it is reused as-is.
func (v *VLLMAdapter) ensureSocket() error {
	if v.socket != nil {
		return nil
	}
	socket, err := zmq.NewSocket(zmq.SUB)
	if err != nil {
		return fmt.Errorf("failed to create ZMQ SUB socket: %w", err)
	}
	v.socket = socket
	v.poller = zmq.NewPoller()
	return nil
}

// getHashAsUint64 converts vLLM hash formats (uint64 or []byte) to uint64.
// This handles both legacy uint64 hashes and new []byte hashes by taking
// the last 8 bytes and interpreting them as a big-endian integer.
func (v *VLLMAdapter) getHashAsUint64(raw any) (uint64, error) {
	switch val := raw.(type) {
	case uint64:
		return val, nil
	case int64:
		// msgpack can decode small integers as int64
		//nolint:gosec // int64 to uint64 conversion is safe here
		return uint64(val), nil
	case []byte:
		if len(val) == 0 {
			return 0, fmt.Errorf("hash byte slice is empty")
		}
		// If the slice is 8 bytes or longer, use the last 8 bytes
		if len(val) >= 8 {
			return binary.BigEndian.Uint64(val[len(val)-8:]), nil
		}
		// If the slice is shorter than 8 bytes, pad it with leading zeros
		padded := make([]byte, 8)
		copy(padded[8-len(val):], val)
		return binary.BigEndian.Uint64(padded), nil
	default:
		return 0, fmt.Errorf("unsupported hash type: %T", val)
	}
}

// vLLM msgpack-specific event structures.
// These structs are designed for msgpack array encoding and match vLLM's format.
type msgpackVLLMEventBatch struct {
	_                struct{} `msgpack:",array"`
	TS               float64
	Events           []msgpack.RawMessage
	DataParallelRank *int `msgpack:",omitempty"`
}

type msgpackVLLMBlockStoredEvent struct {
	_               struct{} `msgpack:",array"`
	Tag             string
	BlockHashes     []any // Changed from []uint64
	ParentBlockHash any   // Changed from *uint64
	TokenIds        []uint32
	BlockSize       int
	LoraID          *int    `msgpack:",omitempty"`
	Medium          *string `msgpack:",omitempty"`
	LoraName        *string `msgpack:",omitempty"`
	// ExtraKeys is present in newer vLLM versions.
	// ExtraKeys any `msgpack:",omitempty"`
}

type msgpackVLLMBlockRemovedEvent struct {
	_           struct{} `msgpack:",array"`
	Tag         string
	BlockHashes []any
	Medium      *string `msgpack:",omitempty"`
}

type msgpackVLLMAllBlocksClearedEvent struct {
	_ struct{} `msgpack:",array"`
}

// ReceiveMessage receives a raw message from the ZMQ socket, parses the vLLM
// 3-part message structure and topic metadata, and returns a RawMessage with
// the payload still in raw bytes. No msgpack decoding happens here.
func (v *VLLMAdapter) ReceiveMessage(ctx context.Context) (*RawMessage, error) {
	// Receive raw message parts from ZMQ socket
	parts, err := v.receiveZMQ(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to receive message: %w", err)
	}

	// Parse vLLM 3-part message structure (topic, sequence, payload)
	// and extract pod ID and model name from the topic in one pass.
	msg, err := parseVLLMMessage(parts)
	if err != nil {
		return nil, err
	}

	// Attach this adapter so the pool worker can call DecodeMessageToEventBatch.
	msg.Adapter = v
	return msg, nil
}

// receiveZMQ blocks until a message is received or context is canceled.
// Returns the raw multi-part ZMQ message as a slice of byte slices.
func (v *VLLMAdapter) receiveZMQ(ctx context.Context) ([][]byte, error) {
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Poll with timeout to allow checking context cancellation
		polled, err := v.poller.Poll(pollTimeout)
		if err != nil {
			return nil, fmt.Errorf("failed to poll ZMQ socket: %w", err)
		}

		if len(polled) == 0 {
			// Timeout, continue to check context
			continue
		}

		parts, err := v.socket.RecvMessageBytes(0)
		if err != nil {
			return nil, fmt.Errorf("failed to receive ZMQ message: %w", err)
		}

		return parts, nil
	}
}

// DecodeMessageToEventBatch decodes the raw msgpack payload of a RawMessage
// into a fully populated EventBatch.
func (v *VLLMAdapter) DecodeMessageToEventBatch(msg *RawMessage) (*events.EventBatch, error) {
	// Decode the payload into vLLM event batch using msgpack
	var vllmBatch msgpackVLLMEventBatch
	if err := msgpack.Unmarshal(msg.Payload, &vllmBatch); err != nil {
		return nil, fmt.Errorf("failed to decode vLLM event batch: %w", err)
	}

	// Convert vLLM events to generic events
	genericEvents := make([]events.GenericEvent, len(vllmBatch.Events))
	for i, rawEventBytes := range vllmBatch.Events {
		genericEvent, err := v.decodeVLLMEvent(rawEventBytes)
		if err != nil {
			return nil, fmt.Errorf("failed to decode vLLM event: %w", err)
		}
		genericEvents[i] = genericEvent
	}

	return &events.EventBatch{
		Timestamp: vllmBatch.TS,
		Events:    genericEvents,
	}, nil
}

// parseVLLMMessage validates and parses a vLLM 3-part message structure,
// including extracting pod ID and model name from the topic.
// vLLM sends messages as: [topic, sequence, payload]
// Returns an error if the message structure is invalid.
func parseVLLMMessage(parts [][]byte) (*RawMessage, error) {
	if len(parts) != 3 {
		return nil, fmt.Errorf("expected 3 message parts from vLLM, got %d", len(parts))
	}

	topic := string(parts[0])
	sequenceBytes := parts[1]
	payload := parts[2]

	// Parse sequence number (8 bytes, big-endian uint64)
	if len(sequenceBytes) != 8 {
		return nil, fmt.Errorf("invalid sequence bytes length: %d", len(sequenceBytes))
	}
	sequence := binary.BigEndian.Uint64(sequenceBytes)

	// Extract pod ID and model name from topic in the same pass
	podID, modelName := parseVLLMTopic(topic)

	return &RawMessage{
		Topic:     topic,
		Sequence:  sequence,
		Payload:   payload,
		PodID:     podID,
		ModelName: modelName,
	}, nil
}

// parseVLLMTopic extracts pod ID and model name from vLLM topic format.
// Expected format: "kv@<pod-id>@<model-name>".
// TODO: Find a way to avoid it.
func parseVLLMTopic(topic string) (string, string) {
	topicParts := strings.Split(topic, "@")
	if len(topicParts) == 3 {
		return topicParts[1], topicParts[2]
	}
	// Fallback if format is unexpected
	return topic, ""
}

// decodeVLLMEvent decodes a single vLLM event using msgpack and converts it to a generic event.
func (v *VLLMAdapter) decodeVLLMEvent(rawEventBytes []byte) (events.GenericEvent, error) {
	// First decode to extract just the tag
	var taggedUnion []any
	if err := msgpack.Unmarshal(rawEventBytes, &taggedUnion); err != nil {
		return nil, fmt.Errorf("failed to decode tagged union: %w", err)
	}

	if len(taggedUnion) < 1 {
		return nil, fmt.Errorf("malformed tagged union: no tag")
	}

	// Extract the event type tag
	tag, ok := taggedUnion[0].(string)
	if !ok {
		return nil, fmt.Errorf("event tag is not a string: %T", taggedUnion[0])
	}

	// Dispatch to appropriate converter
	converter, exists := v.eventConverters[tag]
	if !exists {
		return nil, fmt.Errorf("unknown vLLM event tag: %s", tag)
	}

	return converter(rawEventBytes)
}

// convertBlockStoredEvent decodes and converts a msgpack vLLM BlockStored event to a generic event.
// Parses all hashes from engine-specific formats to uint64.
func (v *VLLMAdapter) convertBlockStoredEvent(rawEventBytes []byte) (events.GenericEvent, error) {
	var vllmEvent msgpackVLLMBlockStoredEvent
	if err := msgpack.Unmarshal(rawEventBytes, &vllmEvent); err != nil {
		return nil, fmt.Errorf("failed to decode BlockStored event: %w", err)
	}

	deviceTier := defaultDeviceTier
	if vllmEvent.Medium != nil {
		deviceTier = strings.ToLower(*vllmEvent.Medium)
	}

	// Parse block hashes
	blockHashes := make([]uint64, 0, len(vllmEvent.BlockHashes))
	for _, rawHash := range vllmEvent.BlockHashes {
		hash, err := v.getHashAsUint64(rawHash)
		if err != nil {
			return nil, fmt.Errorf("failed to parse block hash: %w", err)
		}
		blockHashes = append(blockHashes, hash)
	}

	// Parse parent hash
	var parentHash uint64
	if vllmEvent.ParentBlockHash != nil {
		hash, err := v.getHashAsUint64(vllmEvent.ParentBlockHash)
		if err != nil {
			return nil, fmt.Errorf("failed to parse parent hash: %w", err)
		}
		parentHash = hash
	}

	return &events.BlockStoredEvent{
		BlockHashes: blockHashes,
		Tokens:      vllmEvent.TokenIds,
		ParentHash:  parentHash,
		DeviceTier:  deviceTier,
		LoraID:      vllmEvent.LoraID,
		LoraName:    vllmEvent.LoraName,
	}, nil
}

// convertBlockRemovedEvent decodes and converts a msgpack vLLM BlockRemoved event to a generic event.
// Parses all hashes from engine-specific formats to uint64.
func (v *VLLMAdapter) convertBlockRemovedEvent(rawEventBytes []byte) (events.GenericEvent, error) {
	var vllmEvent msgpackVLLMBlockRemovedEvent
	if err := msgpack.Unmarshal(rawEventBytes, &vllmEvent); err != nil {
		return nil, fmt.Errorf("failed to decode BlockRemoved event: %w", err)
	}

	deviceTier := defaultDeviceTier
	if vllmEvent.Medium != nil {
		deviceTier = strings.ToLower(*vllmEvent.Medium)
	}

	// Parse block hashes
	blockHashes := make([]uint64, 0, len(vllmEvent.BlockHashes))
	for _, rawHash := range vllmEvent.BlockHashes {
		hash, err := v.getHashAsUint64(rawHash)
		if err != nil {
			return nil, fmt.Errorf("failed to parse block hash: %w", err)
		}
		blockHashes = append(blockHashes, hash)
	}

	return &events.BlockRemovedEvent{
		BlockHashes: blockHashes,
		DeviceTier:  deviceTier,
	}, nil
}

// convertAllBlocksClearedEvent converts an AllBlocksCleared event.
func (v *VLLMAdapter) convertAllBlocksClearedEvent(rawEventBytes []byte) (events.GenericEvent, error) {
	return &events.AllBlocksClearedEvent{}, nil
}

// Connect establishes a connection to a remote vLLM endpoint.
func (v *VLLMAdapter) Connect(ctx context.Context, endpoint string) error {
	if err := v.ensureSocket(); err != nil {
		return err
	}
	if err := v.socket.Connect(endpoint); err != nil {
		return fmt.Errorf("failed to connect to endpoint %s: %w", endpoint, err)
	}
	v.poller.Add(v.socket, zmq.POLLIN)
	return nil
}

// Bind listens on a local endpoint for incoming vLLM connections.
func (v *VLLMAdapter) Bind(ctx context.Context, endpoint string) error {
	if err := v.ensureSocket(); err != nil {
		return err
	}
	if err := v.socket.Bind(endpoint); err != nil {
		return fmt.Errorf("failed to bind to endpoint %s: %w", endpoint, err)
	}
	v.poller.Add(v.socket, zmq.POLLIN)
	return nil
}

// SubscribeToTopic sets the topic filter for receiving vLLM messages.
func (v *VLLMAdapter) SubscribeToTopic(topicFilter string) error {
	if err := v.socket.SetSubscribe(topicFilter); err != nil {
		return fmt.Errorf("failed to subscribe to topic filter %s: %w", topicFilter, err)
	}
	return nil
}

// Close closes the ZMQ socket and releases resources.
// Sets socket to nil so ensureSocket() knows to create a fresh one on next use.
func (v *VLLMAdapter) Close() error {
	if v.socket != nil {
		err := v.socket.Close()
		v.socket = nil
		return err
	}
	return nil
}
