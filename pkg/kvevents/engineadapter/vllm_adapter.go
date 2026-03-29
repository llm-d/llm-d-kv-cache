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
		genericEvent, err := decodeEvent(rawEventBytes, v.eventConverters, "vLLM")
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
	_      struct{} `msgpack:",array"`
	Tag    string
	Medium *string `msgpack:",omitempty"`
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

	blockHashes, err := convertBlockHashes(vllmEvent.BlockHashes)
	if err != nil {
		return nil, err
	}

	var parentHash uint64
	if vllmEvent.ParentBlockHash != nil {
		hash, err := getHashAsUint64(vllmEvent.ParentBlockHash)
		if err != nil {
			return nil, fmt.Errorf("failed to parse parent hash: %w", err)
		}
		parentHash = hash
	}

	extraKeys, err := convertExtraKeys(vllmEvent.ExtraKeys)
	if err != nil {
		return nil, err
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

	blockHashes, err := convertBlockHashes(vllmEvent.BlockHashes)
	if err != nil {
		return nil, err
	}

	return &kvevents.BlockRemovedEvent{
		BlockHashes: blockHashes,
		DeviceTier:  deviceTier,
	}, nil
}

// convertAllBlocksClearedEvent converts an AllBlocksCleared event.
func (v *VLLMAdapter) convertAllBlocksClearedEvent(rawEventBytes []byte) (kvevents.GenericEvent, error) {
	var vllmEvent msgpackVLLMAllBlocksClearedEvent
	if err := msgpack.Unmarshal(rawEventBytes, &vllmEvent); err != nil {
		return nil, fmt.Errorf("failed to decode AllBlocksClearedEvent: %w", err)
	}
	deviceTier := ""
	if vllmEvent.Medium != nil {
		deviceTier = *vllmEvent.Medium
	}

	return &kvevents.AllBlocksClearedEvent{
		DeviceTier: deviceTier,
	}, nil
}
