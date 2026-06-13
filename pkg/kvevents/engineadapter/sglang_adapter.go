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
	"github.com/vmihailenco/msgpack/v5/msgpcode"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
)

const (
	// Expected field counts for SGLang msgpack array structs.
	// SGLang uses the same positional wire format as vLLM but may omit trailing optional fields
	// via omit_defaults=True in msgspec.
	// See: sglang/srt/disaggregation/kv_events.py (BlockStored, BlockRemoved classes).
	sglangBlockStoredFieldCount  = 9 // tag + block_hashes + parent + tokens + block_size + lora_id + medium + lora_name + extra_keys
	sglangBlockRemovedFieldCount = 3 // tag + block_hashes + medium

	// Minimum required fields (excluding trailing optional ones).
	sglangBlockStoredMinFields  = 5 // tag + block_hashes + parent + tokens + block_size
	sglangBlockRemovedMinFields = 2 // tag + block_hashes
)

// SGLangAdapter implements the kvevents.EngineAdapter interface for SGLang engines.
// SGLang uses the same msgpack wire format as vLLM but may omit trailing optional fields.
type SGLangAdapter struct {
	eventConverters map[string]func([]byte) (kvevents.GenericEvent, error)
}

// NewSGLangAdapter creates a new SGLang adapter.
func NewSGLangAdapter() *SGLangAdapter {
	adapter := &SGLangAdapter{}

	adapter.eventConverters = map[string]func([]byte) (kvevents.GenericEvent, error){
		eventTagBlockStored:      adapter.convertBlockStoredEvent,
		eventTagBlockRemoved:     adapter.convertBlockRemovedEvent,
		eventTagAllBlocksCleared: adapter.convertAllBlocksClearedEvent,
	}

	return adapter
}

// ShardingKey extracts the pod-id segment from a SGLang raw message topic.
// Expected topic format: "kv@<pod-id>@<model-name>" (same as vLLM).
func (s *SGLangAdapter) ShardingKey(msg *kvevents.RawMessage) string {
	podID, _ := parseTopic(msg.Topic)
	return podID
}

// ParseMessage parses a raw transport message into domain data.
// It extracts pod identity and model name from the topic,
// and decodes the msgpack payload into an EventBatch.
//
//nolint:gocritic // unnamedResult: named returns conflict with nonamedreturns linter
func (s *SGLangAdapter) ParseMessage(msg *kvevents.RawMessage) (string, string, kvevents.EventBatch, error) {
	podID, modelName := parseTopic(msg.Topic)

	var batch msgpackSGLangEventBatch
	if err := msgpack.Unmarshal(msg.Payload, &batch); err != nil {
		return "", "", kvevents.EventBatch{}, fmt.Errorf("failed to decode SGLang event batch: %w", err)
	}

	genericEvents := make([]kvevents.GenericEvent, len(batch.Events))
	for i, rawEventBytes := range batch.Events {
		genericEvent, err := decodeEvent(rawEventBytes, s.eventConverters)
		if err != nil {
			return "", "", kvevents.EventBatch{}, fmt.Errorf("failed to decode SGLang event: %w", err)
		}
		genericEvents[i] = genericEvent
	}

	eventBatch := kvevents.EventBatch{
		Timestamp: batch.TS,
		Events:    genericEvents,
	}

	return podID, modelName, eventBatch, nil
}

// SGLang msgpack event structures.
// These match the vLLM wire format (SGLang uses the same positional encoding).
type msgpackSGLangEventBatch struct {
	_                struct{} `msgpack:",array"`
	TS               float64
	Events           []msgpack.RawMessage
	DataParallelRank *int `msgpack:",omitempty"`
}

type msgpackSGLangBlockStoredEvent struct {
	_               struct{} `msgpack:",array"`
	Tag             string
	BlockHashes     []any
	ParentBlockHash any
	TokenIds        sglangTokenIDs
	BlockSize       int
	LoraID          *int    `msgpack:",omitempty"`
	Medium          *string `msgpack:",omitempty"`
	LoraName        *string `msgpack:",omitempty"`
	ExtraKeys       []any   `msgpack:",omitempty"`
}

// sglangTokenIDs decodes BlockStored.token_ids, which SGLang emits as flat
// [t0, t1, ...] normally or as bigram [[t0,t1],[t1,t2],...] under EAGLE-family
// speculative decoding (see sglang events.py:_record_store_event). The wire
// codes are disjoint (int vs array) so the branch is picked by peeking the
// first element.
//
// The bigram branch flattens N pairs back to N+1 raw tokens. The trailing
// overlap with the next page is dropped by kvblock.chunkTokens as a partial
// block, so the resulting canonical block hashes match the flat-token request
// path -- no bigram awareness is needed in token_processor.go.
type sglangTokenIDs []uint32

func (t *sglangTokenIDs) DecodeMsgpack(dec *msgpack.Decoder) error {
	count, err := dec.DecodeArrayLen()
	if err != nil {
		return err
	}
	if count <= 0 {
		*t = nil
		return nil
	}

	code, err := dec.PeekCode()
	if err != nil {
		return err
	}
	isBigram := msgpcode.IsFixedArray(code) ||
		code == msgpcode.Array16 ||
		code == msgpcode.Array32

	if isBigram {
		out := make([]uint32, count+1)
		if err := decodeBigramTokenIDs(dec, out); err != nil {
			return err
		}
		*t = out
		return nil
	}

	out := make([]uint32, count)
	for i := 0; i < count; i++ {
		v, err := dec.DecodeUint32()
		if err != nil {
			return fmt.Errorf("token_ids[%d]: %w", i, err)
		}
		out[i] = v
	}
	*t = out
	return nil
}

// decodeBigramTokenIDs flattens N overlapping pairs into N+1 raw tokens;
// out must be sized accordingly. Only the first pair contributes its head;
// subsequent pair heads overlap with the previous pair's tail.
func decodeBigramTokenIDs(dec *msgpack.Decoder, out []uint32) error {
	pairs := len(out) - 1
	for i := 0; i < pairs; i++ {
		head, tail, err := decodeBigramPair(dec, i)
		if err != nil {
			return err
		}
		if i == 0 {
			out[0] = head
		}
		out[i+1] = tail
	}
	return nil
}

// decodeBigramPair decodes one [prev, curr] inner array. Extra trailing
// elements (inner > 2) are tolerated; inner < 2 is an error.
func decodeBigramPair(dec *msgpack.Decoder, i int) (uint32, uint32, error) {
	inner, err := dec.DecodeArrayLen()
	if err != nil {
		return 0, 0, fmt.Errorf("token_ids bigram[%d]: %w", i, err)
	}
	if inner < 2 {
		return 0, 0, fmt.Errorf("token_ids bigram[%d]: pair too short, len=%d", i, inner)
	}
	head, err := dec.DecodeUint32()
	if err != nil {
		return 0, 0, fmt.Errorf("token_ids bigram[%d][0]: %w", i, err)
	}
	tail, err := dec.DecodeUint32()
	if err != nil {
		return 0, 0, fmt.Errorf("token_ids bigram[%d][1]: %w", i, err)
	}
	for k := 2; k < inner; k++ {
		if err := dec.Skip(); err != nil {
			return 0, 0, fmt.Errorf("token_ids bigram[%d][%d]: %w", i, k, err)
		}
	}
	return head, tail, nil
}

type msgpackSGLangBlockRemovedEvent struct {
	_           struct{} `msgpack:",array"`
	Tag         string
	BlockHashes []any
	Medium      *string `msgpack:",omitempty"`
}

// padFields pads a msgpack array to the expected field count with nil values.
// Returns the original bytes if already at the expected length, avoiding unnecessary re-marshal overhead.
func padFields(rawEventBytes []byte, fields []any, expectedCount int) ([]byte, error) {
	if len(fields) >= expectedCount {
		return rawEventBytes, nil
	}
	for len(fields) < expectedCount {
		fields = append(fields, nil)
	}
	paddedBytes, err := msgpack.Marshal(fields)
	if err != nil {
		return nil, fmt.Errorf("failed to re-marshal padded event: %w", err)
	}
	return paddedBytes, nil
}

// convertBlockStoredEvent decodes and converts a BlockStored event to a generic event.
// Handles SGLang's shorter arrays by padding missing trailing optional fields with nil.
func (s *SGLangAdapter) convertBlockStoredEvent(rawEventBytes []byte) (kvevents.GenericEvent, error) {
	var fields []any
	if err := msgpack.Unmarshal(rawEventBytes, &fields); err != nil {
		return nil, fmt.Errorf("failed to decode BlockStored event: %w", err)
	}

	if len(fields) < sglangBlockStoredMinFields {
		return nil, fmt.Errorf("BlockStored event has too few fields: %d (minimum %d)", len(fields), sglangBlockStoredMinFields)
	}

	eventBytes, err := padFields(rawEventBytes, fields, sglangBlockStoredFieldCount)
	if err != nil {
		return nil, err
	}

	var event msgpackSGLangBlockStoredEvent
	if err := msgpack.Unmarshal(eventBytes, &event); err != nil {
		return nil, fmt.Errorf("failed to decode BlockStored event: %w", err)
	}

	deviceTier := ""
	if event.Medium != nil {
		deviceTier = *event.Medium
	}

	blockHashes, err := convertBlockHashes(event.BlockHashes)
	if err != nil {
		return nil, err
	}

	var parentHash uint64
	if event.ParentBlockHash != nil {
		hash, err := getHashAsUint64(event.ParentBlockHash)
		if err != nil {
			return nil, fmt.Errorf("failed to parse parent hash: %w", err)
		}
		parentHash = hash
	}

	extraKeys, err := convertExtraKeys(event.ExtraKeys)
	if err != nil {
		return nil, err
	}

	return &kvevents.BlockStoredEvent{
		BlockHashes: blockHashes,
		Tokens:      event.TokenIds,
		ParentHash:  parentHash,
		BlockSize:   event.BlockSize,
		DeviceTier:  deviceTier,
		LoraID:      event.LoraID,
		LoraName:    event.LoraName,
		ExtraKeys:   extraKeys,
	}, nil
}

// convertBlockRemovedEvent decodes and converts a BlockRemoved event to a generic event.
// Handles SGLang's shorter arrays by padding missing trailing optional fields with nil.
func (s *SGLangAdapter) convertBlockRemovedEvent(rawEventBytes []byte) (kvevents.GenericEvent, error) {
	var fields []any
	if err := msgpack.Unmarshal(rawEventBytes, &fields); err != nil {
		return nil, fmt.Errorf("failed to decode BlockRemoved event: %w", err)
	}

	if len(fields) < sglangBlockRemovedMinFields {
		return nil, fmt.Errorf("BlockRemoved event has too few fields: %d (minimum %d)", len(fields), sglangBlockRemovedMinFields)
	}

	eventBytes, err := padFields(rawEventBytes, fields, sglangBlockRemovedFieldCount)
	if err != nil {
		return nil, err
	}

	var event msgpackSGLangBlockRemovedEvent
	if err := msgpack.Unmarshal(eventBytes, &event); err != nil {
		return nil, fmt.Errorf("failed to decode BlockRemoved event: %w", err)
	}

	deviceTier := ""
	if event.Medium != nil {
		deviceTier = *event.Medium
	}

	blockHashes, err := convertBlockHashes(event.BlockHashes)
	if err != nil {
		return nil, err
	}

	return &kvevents.BlockRemovedEvent{
		BlockHashes: blockHashes,
		DeviceTier:  deviceTier,
	}, nil
}

// convertAllBlocksClearedEvent converts an AllBlocksCleared event.
func (s *SGLangAdapter) convertAllBlocksClearedEvent(_ []byte) (kvevents.GenericEvent, error) {
	return &kvevents.AllBlocksClearedEvent{}, nil
}
