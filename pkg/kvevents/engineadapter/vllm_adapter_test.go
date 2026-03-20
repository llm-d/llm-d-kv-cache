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

package engineadapter //nolint:testpackage // Tests access unexported functions

import (
	"testing"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/vmihailenco/msgpack/v5"
)

// TestVLLMShardingKey tests the sharding key extraction from raw messages.
func TestVLLMShardingKey(t *testing.T) {
	adapter := NewVLLMAdapter()
	assert.Equal(t, "pod-123", adapter.ShardingKey(&kvevents.RawMessage{Topic: "kv@pod-123@llama-2-7b"}))
	assert.Equal(t, "fallback", adapter.ShardingKey(&kvevents.RawMessage{Topic: "fallback"}))
}

// TestVLLMParseMessage_Valid tests full message parsing through the adapter.
func TestVLLMParseMessage_Valid(t *testing.T) {
	adapter := NewVLLMAdapter()

	blockStoredEvent := []any{
		"BlockStored",
		[]any{uint64(100), uint64(101)},
		uint64(99),
		[]uint32{1, 2, 3},
		16,
		nil,
		"gpu",
		nil,
		nil,
	}

	batch := []any{
		1234567890.0,
		[]any{blockStoredEvent},
		nil,
	}
	payload, err := msgpack.Marshal(batch)
	require.NoError(t, err)

	msg := &kvevents.RawMessage{
		Topic:    "kv@pod-1@llama-2-7b",
		Sequence: 42,
		Payload:  payload,
	}

	podID, modelName, eventBatch, err := adapter.ParseMessage(msg)
	require.NoError(t, err)
	assert.Equal(t, "pod-1", podID)
	assert.Equal(t, "llama-2-7b", modelName)
	assert.Len(t, eventBatch.Events, 1)

	blockStored, ok := eventBatch.Events[0].(*kvevents.BlockStoredEvent)
	require.True(t, ok)
	assert.Equal(t, []uint64{100, 101}, blockStored.BlockHashes)
	assert.Equal(t, uint64(99), blockStored.ParentHash)
}

// TestVLLMParseMessage_InvalidPayload tests error handling for invalid msgpack data.
func TestVLLMParseMessage_InvalidPayload(t *testing.T) {
	adapter := NewVLLMAdapter()

	msg := &kvevents.RawMessage{
		Topic:   "kv@pod-1@model",
		Payload: []byte{0xFF, 0xFF, 0xFF},
	}

	_, _, _, err := adapter.ParseMessage(msg)
	assert.Error(t, err)
}

// TestVLLMBlockStored tests decoding a valid BlockStored event without LoRA.
func TestVLLMBlockStored(t *testing.T) {
	adapter := NewVLLMAdapter()

	vllmEvent := []any{
		"BlockStored",
		[]any{uint64(100), uint64(101)},
		uint64(99),
		[]uint32{1, 2, 3},
		16,
		nil,
		"gpu",
		nil,
		nil,
	}

	rawBytes, err := msgpack.Marshal(vllmEvent)
	require.NoError(t, err)

	event, err := adapter.decodeVLLMEvent(rawBytes)
	require.NoError(t, err)
	require.NotNil(t, event)

	blockStored, ok := event.(*kvevents.BlockStoredEvent)
	require.True(t, ok, "expected BlockStoredEvent")
	assert.Equal(t, []uint64{100, 101}, blockStored.BlockHashes)
	assert.Equal(t, uint64(99), blockStored.ParentHash)
	assert.Equal(t, []uint32{1, 2, 3}, blockStored.Tokens)
	assert.Equal(t, "gpu", blockStored.DeviceTier)
	assert.Nil(t, blockStored.LoraID)
	assert.Nil(t, blockStored.LoraName)
	assert.Nil(t, blockStored.ExtraKeys)
}

// TestVLLMBlockStoredWithLora tests decoding a valid BlockStored event with LoRA.
func TestVLLMBlockStoredWithLora(t *testing.T) {
	adapter := NewVLLMAdapter()

	vllmEvent := []any{
		"BlockStored",
		[]any{uint64(200), uint64(201)},
		uint64(199),
		[]uint32{4, 5, 6},
		32,
		42,
		"gpu",
		"test-lora",
		[]any{[]any{"uuid-A", "salt"}, nil},
	}

	rawBytes, err := msgpack.Marshal(vllmEvent)
	require.NoError(t, err)

	event, err := adapter.decodeVLLMEvent(rawBytes)
	require.NoError(t, err)
	require.NotNil(t, event)

	blockStored, ok := event.(*kvevents.BlockStoredEvent)
	require.True(t, ok, "expected BlockStoredEvent")
	assert.Equal(t, []uint64{200, 201}, blockStored.BlockHashes)
	assert.Equal(t, uint64(199), blockStored.ParentHash)
	assert.Equal(t, []uint32{4, 5, 6}, blockStored.Tokens)
	assert.Equal(t, "gpu", blockStored.DeviceTier)
	require.NotNil(t, blockStored.LoraID)
	assert.Equal(t, 42, *blockStored.LoraID)
	require.NotNil(t, blockStored.LoraName)
	assert.Equal(t, "test-lora", *blockStored.LoraName)
	require.NotNil(t, blockStored.ExtraKeys)
	assert.Equal(t, [][]any{{"uuid-A", "salt"}, nil}, blockStored.ExtraKeys)
}

// TestDecodeVLLMEvent_BlockStoredMissingTrailingFields tests backward compatibility
// when trailing optional fields are absent (older vLLM with omit_defaults=True).
func TestDecodeVLLMEvent_BlockStoredMissingTrailingFields(t *testing.T) {
	adapter := NewVLLMAdapter()

	tests := []struct {
		name       string
		event      []any
		wantLoraID *int
		wantMedium string
		wantLora   *string
	}{
		{
			name: "missing lora_name only",
			event: []any{
				"BlockStored",
				[]any{uint64(300), uint64(301)},
				uint64(299),
				[]uint32{7, 8, 9},
				64,
				123,
				"gpu",
			},
			wantLoraID: intPtr(123),
			wantMedium: "gpu",
			wantLora:   nil,
		},
		{
			name: "missing medium and lora_name",
			event: []any{
				"BlockStored",
				[]any{uint64(300)},
				uint64(299),
				[]uint32{7, 8, 9},
				64,
				42,
			},
			wantLoraID: intPtr(42),
			wantMedium: "",
			wantLora:   nil,
		},
		{
			name: "only required fields",
			event: []any{
				"BlockStored",
				[]any{uint64(300)},
				uint64(299),
				[]uint32{7, 8, 9},
				64,
			},
			wantLoraID: nil,
			wantMedium: "",
			wantLora:   nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rawBytes, err := msgpack.Marshal(tt.event)
			require.NoError(t, err)

			event, err := adapter.decodeVLLMEvent(rawBytes)
			require.NoError(t, err)

			blockStored, ok := event.(*kvevents.BlockStoredEvent)
			require.True(t, ok)
			assert.Equal(t, tt.wantLoraID, blockStored.LoraID)
			assert.Equal(t, tt.wantMedium, blockStored.DeviceTier)
			assert.Equal(t, tt.wantLora, blockStored.LoraName)
		})
	}
}

// TestDecodeVLLMEvent_BlockStoredExtraTrailingFields tests forward compatibility
// when newer vLLM sends fields this consumer doesn't know about.
func TestDecodeVLLMEvent_BlockStoredExtraTrailingFields(t *testing.T) {
	adapter := NewVLLMAdapter()

	// Simulate a future vLLM version with extra_keys and another unknown field
	vllmEvent := []any{
		"BlockStored",
		[]any{uint64(400), uint64(401)},
		uint64(399),
		[]uint32{10, 11, 12},
		16,
		nil,
		"gpu",
		"my-lora",
		[]any{[]any{"extra", "keys"}}, // [8] extra_keys
		"completely-unknown-field",    // [9] future unknown — silently ignored
	}

	rawBytes, err := msgpack.Marshal(vllmEvent)
	require.NoError(t, err)

	event, err := adapter.decodeVLLMEvent(rawBytes)
	require.NoError(t, err)

	blockStored, ok := event.(*kvevents.BlockStoredEvent)
	require.True(t, ok)
	assert.Equal(t, []uint64{400, 401}, blockStored.BlockHashes)
	assert.Equal(t, uint64(399), blockStored.ParentHash)
	assert.Equal(t, []uint32{10, 11, 12}, blockStored.Tokens)
	assert.Equal(t, "gpu", blockStored.DeviceTier)
	assert.Nil(t, blockStored.LoraID)
	require.NotNil(t, blockStored.LoraName)
	assert.Equal(t, "my-lora", *blockStored.LoraName)
	require.NotNil(t, blockStored.ExtraKeys)
	assert.Equal(t, [][]any{{"extra", "keys"}}, blockStored.ExtraKeys)
}

// TestDecodeVLLMEvent_BlockRemovedExtraTrailingFields tests forward compatibility for BlockRemoved.
func TestDecodeVLLMEvent_BlockRemovedExtraTrailingFields(t *testing.T) {
	adapter := NewVLLMAdapter()

	vllmEvent := []any{
		"BlockRemoved",
		[]any{uint64(500)},
		"cpu",
		"future-field-1",
		"future-field-2",
	}

	rawBytes, err := msgpack.Marshal(vllmEvent)
	require.NoError(t, err)

	event, err := adapter.decodeVLLMEvent(rawBytes)
	require.NoError(t, err)

	blockRemoved, ok := event.(*kvevents.BlockRemovedEvent)
	require.True(t, ok)
	assert.Equal(t, []uint64{500}, blockRemoved.BlockHashes)
	assert.Equal(t, "cpu", blockRemoved.DeviceTier)
}

// TestDecodeVLLMEvent_BlockRemovedMissingMedium tests backward compat for BlockRemoved.
func TestDecodeVLLMEvent_BlockRemovedMissingMedium(t *testing.T) {
	adapter := NewVLLMAdapter()

	vllmEvent := []any{
		"BlockRemoved",
		[]any{uint64(600)},
	}

	rawBytes, err := msgpack.Marshal(vllmEvent)
	require.NoError(t, err)

	event, err := adapter.decodeVLLMEvent(rawBytes)
	require.NoError(t, err)

	blockRemoved, ok := event.(*kvevents.BlockRemovedEvent)
	require.True(t, ok)
	assert.Equal(t, []uint64{600}, blockRemoved.BlockHashes)
	assert.Equal(t, "", blockRemoved.DeviceTier)
}

func intPtr(v int) *int {
	return &v
}

// TestVLLMBlockStoredInvalidExtraKeys tests invalid extra_keys type.
func TestVLLMBlockStoredInvalidExtraKeys(t *testing.T) {
	adapter := NewVLLMAdapter()

	vllmEvent := []any{
		"BlockStored",
		[]any{uint64(100)},
		uint64(99),
		[]uint32{1, 2},
		16,
		nil,
		"gpu",
		nil,
		[]any{"invalid_string"},
	}

	rawBytes, err := msgpack.Marshal(vllmEvent)
	require.NoError(t, err)

	_, err = adapter.decodeVLLMEvent(rawBytes)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "extra_keys[0] has invalid type")
}

// TestVLLMBlockRemoved tests decoding a valid BlockRemoved event.
func TestVLLMBlockRemoved(t *testing.T) {
	adapter := NewVLLMAdapter()

	medium := "cpu"
	vllmEvent := []any{
		"BlockRemoved",
		[]any{uint64(200), uint64(201), uint64(202)},
		&medium,
	}

	rawBytes, err := msgpack.Marshal(vllmEvent)
	require.NoError(t, err)

	event, err := adapter.decodeVLLMEvent(rawBytes)
	require.NoError(t, err)
	require.NotNil(t, event)

	blockRemoved, ok := event.(*kvevents.BlockRemovedEvent)
	require.True(t, ok, "expected BlockRemovedEvent")
	assert.Equal(t, []uint64{200, 201, 202}, blockRemoved.BlockHashes)
	assert.Equal(t, "cpu", blockRemoved.DeviceTier)
}

// TestVLLMAllBlocksCleared tests decoding a valid AllBlocksCleared event.
func TestVLLMAllBlocksCleared(t *testing.T) {
	adapter := NewVLLMAdapter()

	vllmEvent := []any{"AllBlocksCleared"}

	rawBytes, err := msgpack.Marshal(vllmEvent)
	require.NoError(t, err)

	event, err := adapter.decodeVLLMEvent(rawBytes)
	require.NoError(t, err)
	require.NotNil(t, event)

	_, ok := event.(*kvevents.AllBlocksClearedEvent)
	require.True(t, ok, "expected AllBlocksClearedEvent")
}

// TestVLLMUnknownTag tests error handling for unknown event tags.
func TestVLLMUnknownTag(t *testing.T) {
	adapter := NewVLLMAdapter()

	vllmEvent := []any{"UnknownEventType", "some", "data"}

	rawBytes, err := msgpack.Marshal(vllmEvent)
	require.NoError(t, err)

	event, err := adapter.decodeVLLMEvent(rawBytes)
	assert.Error(t, err)
	assert.Nil(t, event)
	assert.Contains(t, err.Error(), "unknown vLLM event tag")
}

// TestVLLMMalformedPayload tests error handling for malformed msgpack data.
func TestVLLMMalformedPayload(t *testing.T) {
	adapter := NewVLLMAdapter()

	rawBytes := []byte{0xFF, 0xFF, 0xFF}

	event, err := adapter.decodeVLLMEvent(rawBytes)
	assert.Error(t, err)
	assert.Nil(t, event)
}

// TestVLLMEmptyPayload tests error handling for empty event bytes.
func TestVLLMEmptyPayload(t *testing.T) {
	adapter := NewVLLMAdapter()

	rawBytes := []byte{}

	event, err := adapter.decodeVLLMEvent(rawBytes)
	assert.Error(t, err)
	assert.Nil(t, event)
}

// TestVLLMMissingTag tests error handling for events without a tag.
func TestVLLMMissingTag(t *testing.T) {
	adapter := NewVLLMAdapter()

	vllmEvent := []any{}

	rawBytes, err := msgpack.Marshal(vllmEvent)
	require.NoError(t, err)

	event, err := adapter.decodeVLLMEvent(rawBytes)
	assert.Error(t, err)
	assert.Nil(t, event)
	assert.Contains(t, err.Error(), "malformed tagged union")
}

// TestVLLMEventBatch_NestedArrayEvents tests batch decoding with nested msgpack arrays.
func TestVLLMEventBatch_NestedArrayEvents(t *testing.T) {
	adapter := NewVLLMAdapter()

	blockStoredEvent := []any{
		"BlockStored",
		[]any{uint64(10), uint64(11)},
		uint64(9),
		[]uint32{1, 2, 3},
		16,
		nil,
		"gpu",
		nil,
		nil,
	}

	batch := []any{
		1234567890.0,
		[]any{blockStoredEvent},
		nil,
	}

	payload, err := msgpack.Marshal(batch)
	require.NoError(t, err)

	msg := &kvevents.RawMessage{
		Topic:    "kv@pod-1@model",
		Sequence: 1,
		Payload:  payload,
	}

	_, _, eventBatch, err := adapter.ParseMessage(msg)
	require.NoError(t, err)
	require.Len(t, eventBatch.Events, 1)

	blockStored, ok := eventBatch.Events[0].(*kvevents.BlockStoredEvent)
	require.True(t, ok, "expected BlockStoredEvent")
	assert.Equal(t, []uint64{10, 11}, blockStored.BlockHashes)
	assert.Equal(t, uint64(9), blockStored.ParentHash)
	assert.Equal(t, []uint32{1, 2, 3}, blockStored.Tokens)
	assert.Equal(t, "gpu", blockStored.DeviceTier)
}
