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
	"fmt"

	"github.com/vmihailenco/msgpack/v5"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
)

// VLLMAdapter implements the kvevents.EngineAdapter interface for vLLM engines.
// It parses raw transport messages (topic + msgpack payload) into domain events.
//
// vLLM serializes events using msgspec with array_like=True and omit_defaults=True,
// producing positional msgpack arrays where trailing default fields may be absent.
// To maintain forward and backward compatibility across vLLM versions (new fields
// appended or trailing fields omitted), we decode into []any and extract fields
// positionally with length guards instead of using fixed structs.
type VLLMAdapter struct {
	eventConverters map[string]func([]any) (kvevents.GenericEvent, error)
}

// NewVLLMAdapter creates a new vLLM adapter.
func NewVLLMAdapter() *VLLMAdapter {
	adapter := &VLLMAdapter{}

	adapter.eventConverters = map[string]func([]any) (kvevents.GenericEvent, error){
		eventTagBlockStored:      adapter.convertBlockStoredEvent,
		eventTagBlockRemoved:     adapter.convertBlockRemovedEvent,
		eventTagAllBlocksCleared: adapter.convertAllBlocksClearedEvent,
	}

	return adapter
}

// ShardingKey extracts the pod-id segment from a vLLM raw message topic.
// Expected topic format: "kv@<pod-id>@<model-name>".
func (v *VLLMAdapter) ShardingKey(msg *kvevents.RawMessage) string {
	podID, _ := parseTopic(msg.Topic)
	return podID
}

// ParseMessage parses a raw transport message into domain data.
// It extracts pod identity and model name from the topic,
// and decodes the msgpack payload into an EventBatch.
//
//nolint:gocritic // unnamedResult: named returns conflict with nonamedreturns linter
func (v *VLLMAdapter) ParseMessage(msg *kvevents.RawMessage) (string, string, kvevents.EventBatch, error) {
	podID, modelName := parseTopic(msg.Topic)

	var vllmBatch msgpackVLLMEventBatch
	if err := msgpack.Unmarshal(msg.Payload, &vllmBatch); err != nil {
		return "", "", kvevents.EventBatch{}, fmt.Errorf("failed to decode vLLM event batch: %w", err)
	}

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

// vLLM msgpack event batch structure.
// This struct uses array encoding to match vLLM's msgspec array_like=True format.
type msgpackVLLMEventBatch struct {
	_                struct{} `msgpack:",array"`
	TS               float64
	Events           []msgpack.RawMessage
	DataParallelRank *int `msgpack:",omitempty"`
}

// decodeVLLMEvent decodes a single vLLM event from msgpack bytes into a domain event.
// It performs a single unmarshal into []any and passes the decoded fields to the
// appropriate converter, avoiding double-decode overhead.
func (v *VLLMAdapter) decodeVLLMEvent(rawEventBytes []byte) (kvevents.GenericEvent, error) {
	var fields []any
	if err := msgpack.Unmarshal(rawEventBytes, &fields); err != nil {
		return nil, fmt.Errorf("failed to decode tagged union: %w", err)
	}

	if len(fields) < 1 {
		return nil, fmt.Errorf("malformed tagged union: no tag")
	}

	tag, ok := fields[0].(string)
	if !ok {
		return nil, fmt.Errorf("event tag is not a string: %T", fields[0])
	}

	converter, exists := v.eventConverters[tag]
	if !exists {
		return nil, fmt.Errorf("unknown vLLM event tag: %s", tag)
	}

	return converter(fields)
}

// fieldAt returns the element at index i from fields, or nil if out of bounds.
func fieldAt(fields []any, i int) any {
	if i < len(fields) {
		return fields[i]
	}
	return nil
}

// convertBlockStoredEvent converts a decoded []any into a BlockStoredEvent.
// vLLM field positions (array_like=True, tag=True):
//
//	[0] tag           string            (consumed by decodeVLLMEvent)
//	[1] block_hashes  []hash
//	[2] parent_hash   hash|nil
//	[3] token_ids     []uint32
//	[4] block_size    int               (consumed but not stored)
//	[5] lora_id       int|nil           (optional, omit_defaults)
//	[6] medium        string|nil        (optional, omit_defaults)
//	[7] lora_name     string|nil        (optional, omit_defaults)
//	[8] extra_keys    [][]any|nil       (optional, omit_defaults)
//
// Trailing fields may be absent in older vLLM versions. Extra trailing fields
// from newer vLLM versions are silently ignored.
func (v *VLLMAdapter) convertBlockStoredEvent(fields []any) (kvevents.GenericEvent, error) {
	// Positions 0-4 are required (tag + 4 data fields).
	if len(fields) < 5 {
		return nil, fmt.Errorf("BlockStored: need at least 5 fields, got %d", len(fields))
	}

	// [1] block_hashes
	rawHashes, ok := fields[1].([]any)
	if !ok {
		return nil, fmt.Errorf("BlockStored: block_hashes is not an array: %T", fields[1])
	}
	blockHashes, err := convertBlockHashes(rawHashes)
	if err != nil {
		return nil, err
	}

	// [2] parent_hash
	var parentHash uint64
	if fields[2] != nil {
		hash, err := getHashAsUint64(fields[2])
		if err != nil {
			return nil, fmt.Errorf("failed to parse parent hash: %w", err)
		}
		parentHash = hash
	}

	// [3] token_ids
	tokens, err := toUint32Slice(fields[3])
	if err != nil {
		return nil, fmt.Errorf("BlockStored: %w", err)
	}

	// [4] block_size — consumed but not stored in domain event

	// [5] lora_id (optional)
	var loraID *int
	if raw := fieldAt(fields, 5); raw != nil {
		id, err := toInt(raw)
		if err != nil {
			return nil, fmt.Errorf("BlockStored: lora_id: %w", err)
		}
		loraID = &id
	}

	// [6] medium / device tier (optional)
	var deviceTier string
	if raw := fieldAt(fields, 6); raw != nil {
		s, ok := raw.(string)
		if !ok {
			return nil, fmt.Errorf("BlockStored: medium is not a string: %T", raw)
		}
		deviceTier = s
	}

	// [7] lora_name (optional)
	var loraName *string
	if raw := fieldAt(fields, 7); raw != nil {
		s, ok := raw.(string)
		if !ok {
			return nil, fmt.Errorf("BlockStored: lora_name is not a string: %T", raw)
		}
		loraName = &s
	}

	// [8] extra_keys (optional)
	var extraKeys [][]any
	if raw := fieldAt(fields, 8); raw != nil {
		rawSlice, ok := raw.([]any)
		if !ok {
			return nil, fmt.Errorf("BlockStored: extra_keys is not an array: %T", raw)
		}
		extraKeys, err = convertExtraKeys(rawSlice)
		if err != nil {
			return nil, err
		}
	}

	return &kvevents.BlockStoredEvent{
		BlockHashes: blockHashes,
		Tokens:      tokens,
		ParentHash:  parentHash,
		DeviceTier:  deviceTier,
		LoraID:      loraID,
		LoraName:    loraName,
		ExtraKeys:   extraKeys,
	}, nil
}

// convertBlockRemovedEvent converts a decoded []any into a BlockRemovedEvent.
// vLLM field positions:
//
//	[0] tag           string
//	[1] block_hashes  []hash
//	[2] medium        string|nil      (optional, omit_defaults)
func (v *VLLMAdapter) convertBlockRemovedEvent(fields []any) (kvevents.GenericEvent, error) {
	if len(fields) < 2 {
		return nil, fmt.Errorf("BlockRemoved: need at least 2 fields, got %d", len(fields))
	}

	rawHashes, ok := fields[1].([]any)
	if !ok {
		return nil, fmt.Errorf("BlockRemoved: block_hashes is not an array: %T", fields[1])
	}
	blockHashes, err := convertBlockHashes(rawHashes)
	if err != nil {
		return nil, err
	}

	var deviceTier string
	if raw := fieldAt(fields, 2); raw != nil {
		s, ok := raw.(string)
		if !ok {
			return nil, fmt.Errorf("BlockRemoved: medium is not a string: %T", raw)
		}
		deviceTier = s
	}

	return &kvevents.BlockRemovedEvent{
		BlockHashes: blockHashes,
		DeviceTier:  deviceTier,
	}, nil
}

// convertAllBlocksClearedEvent converts a decoded []any into an AllBlocksClearedEvent.
func (v *VLLMAdapter) convertAllBlocksClearedEvent(_ []any) (kvevents.GenericEvent, error) {
	return &kvevents.AllBlocksClearedEvent{}, nil
}

// toUint32Slice converts a msgpack-decoded []any of integers to []uint32.
func toUint32Slice(raw any) ([]uint32, error) {
	arr, ok := raw.([]any)
	if !ok {
		return nil, fmt.Errorf("token_ids is not an array: %T", raw)
	}
	result := make([]uint32, len(arr))
	for i, v := range arr {
		n, err := toInt(v)
		if err != nil {
			return nil, fmt.Errorf("token_ids[%d]: %w", i, err)
		}
		//nolint:gosec // token IDs fit in uint32
		result[i] = uint32(n)
	}
	return result, nil
}

// toInt converts a msgpack-decoded numeric value to int.
func toInt(raw any) (int, error) {
	switch v := raw.(type) {
	case int64:
		return int(v), nil
	case uint64:
		//nolint:gosec // token IDs and lora IDs fit in int; overflow is not a concern here
		return int(v), nil
	case int8:
		return int(v), nil
	case int16:
		return int(v), nil
	case int32:
		return int(v), nil
	case uint8:
		return int(v), nil
	case uint16:
		return int(v), nil
	case uint32:
		return int(v), nil
	default:
		return 0, fmt.Errorf("unsupported numeric type: %T", raw)
	}
}
