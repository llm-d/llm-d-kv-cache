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
	"encoding/binary"
	"fmt"
	"strings"

	"github.com/vmihailenco/msgpack/v5"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
)

const (
	// vLLM event type tags.
	eventTagBlockStored      = "BlockStored"
	eventTagBlockRemoved     = "BlockRemoved"
	eventTagAllBlocksCleared = "AllBlocksCleared"
)

// VLLMAdapter implements the kvevents.EngineAdapter interface for vLLM engines.
// It parses raw transport messages (topic + msgpack payload) into domain events.
type VLLMAdapter struct {
	eventConverters map[string]func([]byte) (kvevents.GenericEvent, error)
}

// NewVLLMAdapter creates a new vLLM adapter.
func NewVLLMAdapter() *VLLMAdapter {
	adapter := &VLLMAdapter{}

	adapter.eventConverters = map[string]func([]byte) (kvevents.GenericEvent, error){
		eventTagBlockStored:      adapter.convertBlockStoredEvent,
		eventTagBlockRemoved:     adapter.convertBlockRemovedEvent,
		eventTagAllBlocksCleared: adapter.convertAllBlocksClearedEvent,
	}

	return adapter
}

// ShardingKey extracts the pod-id segment from a vLLM raw message topic.
// Expected topic format: "kv@<pod-id>@<model-name>".
func (v *VLLMAdapter) ShardingKey(msg *kvevents.RawMessage) string {
	podID, _ := parseVLLMTopic(msg.Topic)
	return podID
}

// ParseMessage parses a raw transport message into domain data.
// It extracts pod identity and model name from the topic,
// and decodes the msgpack payload into an EventBatch.
//
//nolint:gocritic // unnamedResult: named returns conflict with nonamedreturns linter
func (v *VLLMAdapter) ParseMessage(msg *kvevents.RawMessage) (string, string, kvevents.EventBatch, error) {
	// Extract pod ID and model name from topic
	podID, modelName := parseVLLMTopic(msg.Topic)

	// Decode the payload into vLLM event batch using msgpack
	var vllmBatch msgpackVLLMEventBatch
	if err := msgpack.Unmarshal(msg.Payload, &vllmBatch); err != nil {
		return "", "", kvevents.EventBatch{}, fmt.Errorf("failed to decode vLLM event batch: %w", err)
	}

	// Convert vLLM events to generic events
	genericEvents := make([]kvevents.GenericEvent, len(vllmBatch.Events))
	for i, rawEventBytes := range vllmBatch.Events {
		genericEvent, err := v.decodeVLLMEvent(rawEventBytes)
		if err != nil {
			return "", "", kvevents.EventBatch{}, fmt.Errorf("failed to decode vLLM event: %w", err)
		}
		genericEvents[i] = genericEvent
	}

	batch := kvevents.EventBatch{
		Timestamp: vllmBatch.TS,
		Events:    genericEvents,
	}

	return podID, modelName, batch, nil
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
		if len(val) >= 8 {
			return binary.BigEndian.Uint64(val[len(val)-8:]), nil
		}
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
	BlockHashes     []any
	ParentBlockHash any
	TokenIds        []uint32
	BlockSize       int
	LoraID          *int    `msgpack:",omitempty"`
	Medium          *string `msgpack:",omitempty"`
	LoraName        *string `msgpack:",omitempty"`
	ExtraKeys       []any   `msgpack:",omitempty"`
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

// parseVLLMTopic extracts pod ID and model name from vLLM topic format.
// Expected format: "kv@<pod-id>@<model-name>".
//
//nolint:gocritic // unnamedResult: named returns conflict with nonamedreturns linter
func parseVLLMTopic(topic string) (string, string) {
	topicParts := strings.Split(topic, "@")
	if len(topicParts) == 3 {
		return topicParts[1], topicParts[2]
	}
	return topic, ""
}

// decodeVLLMEvent decodes a single vLLM event using msgpack and converts it to a generic event.
func (v *VLLMAdapter) decodeVLLMEvent(rawEventBytes []byte) (kvevents.GenericEvent, error) {
	// First decode to extract just the tag
	var taggedUnion []any
	if err := msgpack.Unmarshal(rawEventBytes, &taggedUnion); err != nil {
		return nil, fmt.Errorf("failed to decode tagged union: %w", err)
	}

	if len(taggedUnion) < 1 {
		return nil, fmt.Errorf("malformed tagged union: no tag")
	}

	tag, ok := taggedUnion[0].(string)
	if !ok {
		return nil, fmt.Errorf("event tag is not a string: %T", taggedUnion[0])
	}

	converter, exists := v.eventConverters[tag]
	if !exists {
		return nil, fmt.Errorf("unknown vLLM event tag: %s", tag)
	}

	return converter(rawEventBytes)
}

// convertBlockStoredEvent decodes and converts a msgpack vLLM BlockStored event to a generic event.
func (v *VLLMAdapter) convertBlockStoredEvent(rawEventBytes []byte) (kvevents.GenericEvent, error) {
	var vllmEvent msgpackVLLMBlockStoredEvent
	if err := msgpack.Unmarshal(rawEventBytes, &vllmEvent); err != nil {
		return nil, fmt.Errorf("failed to decode BlockStored event: %w", err)
	}

	deviceTier := ""
	if vllmEvent.Medium != nil {
		deviceTier = *vllmEvent.Medium
	}

	blockHashes := make([]uint64, 0, len(vllmEvent.BlockHashes))
	for _, rawHash := range vllmEvent.BlockHashes {
		hash, err := v.getHashAsUint64(rawHash)
		if err != nil {
			return nil, fmt.Errorf("failed to parse block hash: %w", err)
		}
		blockHashes = append(blockHashes, hash)
	}

	var parentHash uint64
	if vllmEvent.ParentBlockHash != nil {
		hash, err := v.getHashAsUint64(vllmEvent.ParentBlockHash)
		if err != nil {
			return nil, fmt.Errorf("failed to parse parent hash: %w", err)
		}
		parentHash = hash
	}

	// Convert extra_keys if present
	var extraKeys [][]any
	if vllmEvent.ExtraKeys != nil {
		extraKeys = make([][]any, 0, len(vllmEvent.ExtraKeys))
		for i, rawKey := range vllmEvent.ExtraKeys {
			if rawKey == nil {
				extraKeys = append(extraKeys, nil)
			} else if keySlice, ok := rawKey.([]any); ok {
				extraKeys = append(extraKeys, keySlice)
			} else {
				return nil, fmt.Errorf("extra_keys[%d] has invalid type %T, expected []any or nil", i, rawKey)
			}
		}
	}

	return &kvevents.BlockStoredEvent{
		BlockHashes: blockHashes,
		Tokens:      vllmEvent.TokenIds,
		ParentHash:  parentHash,
		DeviceTier:  deviceTier,
		LoraID:      vllmEvent.LoraID,
		LoraName:    vllmEvent.LoraName,
		ExtraKeys:   extraKeys,
	}, nil
}

// convertBlockRemovedEvent decodes and converts a msgpack vLLM BlockRemoved event to a generic event.
func (v *VLLMAdapter) convertBlockRemovedEvent(rawEventBytes []byte) (kvevents.GenericEvent, error) {
	var vllmEvent msgpackVLLMBlockRemovedEvent
	if err := msgpack.Unmarshal(rawEventBytes, &vllmEvent); err != nil {
		return nil, fmt.Errorf("failed to decode BlockRemoved event: %w", err)
	}

	deviceTier := ""
	if vllmEvent.Medium != nil {
		deviceTier = *vllmEvent.Medium
	}

	blockHashes := make([]uint64, 0, len(vllmEvent.BlockHashes))
	for _, rawHash := range vllmEvent.BlockHashes {
		hash, err := v.getHashAsUint64(rawHash)
		if err != nil {
			return nil, fmt.Errorf("failed to parse block hash: %w", err)
		}
		blockHashes = append(blockHashes, hash)
	}

	return &kvevents.BlockRemovedEvent{
		BlockHashes: blockHashes,
		DeviceTier:  deviceTier,
	}, nil
}

// convertAllBlocksClearedEvent converts an AllBlocksCleared event.
func (v *VLLMAdapter) convertAllBlocksClearedEvent(_ []byte) (kvevents.GenericEvent, error) {
	return &kvevents.AllBlocksClearedEvent{}, nil
}
