/*
Copyright 2025 The llm-d Authors.

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

	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents/decoder"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents/events"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents/transport"
	"github.com/vmihailenco/msgpack/v5"
)

const (
	// vLLM event type tags
	eventTagBlockStored      = "BlockStored"
	eventTagBlockRemoved     = "BlockRemoved"
	eventTagAllBlocksCleared = "AllBlocksCleared"

	defaultDeviceTier = "gpu"
)

// VLLMAdapter implements the EngineAdapter interface for vLLM engines.
type VLLMAdapter struct {
	transport       transport.Transport
	decoder         decoder.Decoder
	eventConverters map[string]func([]byte) (events.GenericEvent, error)
}

// NewVLLMAdapter creates a new vLLM adapter with ZMQ transport and msgpack decoder.
// Returns an error if the transport cannot be created.
func NewVLLMAdapter() (*VLLMAdapter, error) {
	trans, err := transport.NewZMQTransport()
	if err != nil {
		return nil, fmt.Errorf("failed to create ZMQ transport: %w", err)
	}

	adapter := &VLLMAdapter{
		transport: trans,
		decoder:   decoder.NewMsgpackDecoder(),
	}

	// Initialize event converters map
	adapter.eventConverters = map[string]func([]byte) (events.GenericEvent, error){
		eventTagBlockStored:      adapter.convertBlockStoredEvent,
		eventTagBlockRemoved:     adapter.convertBlockRemovedEvent,
		eventTagAllBlocksCleared: adapter.convertAllBlocksClearedEvent,
	}

	return adapter, nil
}

// Transport returns the Transport.
func (v *VLLMAdapter) Transport() transport.Transport {
	return v.transport
}

// Decoder returns the Decoder.
func (v *VLLMAdapter) Decoder() decoder.Decoder {
	return v.decoder
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

// vLLM msgpack-specific event structures
// These structs are designed for msgpack array encoding and match vLLM's format
type msgpackVLLMEventBatch struct {
	_                struct{} `msgpack:",array"`
	TS               float64
	Events           [][]byte // Raw event bytes (decoder-agnostic)
	DataParallelRank *int     `msgpack:",omitempty"`
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

// vllmMessage represents a parsed vLLM 3-part message.
type vllmMessage struct {
	Topic    string
	Sequence uint64
	Payload  []byte
}

// ReceiveAndDecode receives a message from the transport, parses the vLLM
// 3-part message structure, decodes the payload using the decoder, and returns
// a batch of generic events.
func (v *VLLMAdapter) ReceiveAndDecode(ctx context.Context) (*events.EventBatch, error) {
	// Receive raw message parts from transport
	parts, err := v.transport.Receive(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to receive message: %w", err)
	}

	// Parse vLLM 3-part message structure
	msg, err := parseVLLMMessage(parts)
	if err != nil {
		return nil, err
	}

	// Extract metadata from topic
	podID, modelName := parseVLLMTopic(msg.Topic)

	// Decode the payload into vLLM event batch using the decoder
	var vllmBatch msgpackVLLMEventBatch
	if err := v.decoder.Decode(msg.Payload, &vllmBatch); err != nil {
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
		Metadata: events.Metadata{
			Topic:     msg.Topic,
			PodID:     podID,
			ModelName: modelName,
			Sequence:  msg.Sequence,
			Engine:    "vllm",
		},
		Timestamp: vllmBatch.TS,
		Events:    genericEvents,
	}, nil
}

// parseVLLMMessage validates and parses a vLLM 3-part message structure.
// vLLM sends messages as: [topic, sequence, payload]
// Returns an error if the message structure is invalid.
func parseVLLMMessage(parts [][]byte) (*vllmMessage, error) {
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

	return &vllmMessage{
		Topic:    topic,
		Sequence: sequence,
		Payload:  payload,
	}, nil
}

// parseVLLMTopic extracts pod ID and model name from vLLM topic format.
// Expected format: "pod_id@model_name"
// TODO: Find a way to avoid it
func parseVLLMTopic(topic string) (podID, modelName string) {
	parts := strings.SplitN(topic, "@", 2)
	if len(parts) == 2 {
		return parts[0], parts[1]
	}
	// Fallback if format is unexpected
	return topic, ""
}

// decodeVLLMEvent decodes a single vLLM event using the decoder and converts it to a generic event.
// vLLM events are tagged unions: [tag, ...fields]
func (v *VLLMAdapter) decodeVLLMEvent(rawEventBytes []byte) (events.GenericEvent, error) {
	// First decode to extract just the tag
	var taggedUnion []any
	if err := v.decoder.Decode(rawEventBytes, &taggedUnion); err != nil {
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

// TODO: not sure if it best to keep or remove these

// Connect establishes a connection to a remote vLLM endpoint.
func (v *VLLMAdapter) Connect(ctx context.Context, endpoint string) error {
	return v.transport.Connect(ctx, endpoint)
}

// Bind listens on a local endpoint for incoming vLLM connections.
func (v *VLLMAdapter) Bind(ctx context.Context, endpoint string) error {
	return v.transport.Bind(ctx, endpoint)
}

// SubscribeToTopic sets the topic filter for receiving vLLM messages.
func (v *VLLMAdapter) SubscribeToTopic(topicFilter string) error {
	return v.transport.Subscribe(topicFilter)
}

// Close closes the adapter and releases all resources.
func (v *VLLMAdapter) Close() error {
	return v.transport.Close()
}
