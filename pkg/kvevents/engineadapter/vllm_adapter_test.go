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

package engineadapter //nolint:testpackage // Tests access unexported functions parseVLLMTopic and decodeVLLMEvent

import (
	"encoding/binary"
	"testing"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/vmihailenco/msgpack/v5"
)

// TestParseVLLMTopic tests topic parsing.
func TestParseVLLMTopic_Valid(t *testing.T) {
	podID, modelName := parseVLLMTopic("kv@pod-123@llama-2-7b")
	assert.Equal(t, "pod-123", podID)
	assert.Equal(t, "llama-2-7b", modelName)
}

func TestParseVLLMTopic_NoModel(t *testing.T) {
	podID, modelName := parseVLLMTopic("pod-123@llama-2-7b")
	// Only 2 parts, falls through to default
	assert.Equal(t, "pod-123@llama-2-7b", podID)
	assert.Equal(t, "", modelName)
}

// TestShardingKey tests the sharding key extraction from raw messages.
func TestShardingKey(t *testing.T) {
	adapter := NewVLLMAdapter()
	assert.Equal(t, "pod-123", adapter.ShardingKey(&kvevents.RawMessage{Topic: "kv@pod-123@llama-2-7b"}))
	assert.Equal(t, "fallback", adapter.ShardingKey(&kvevents.RawMessage{Topic: "fallback"}))
}

// TestParseMessage_Valid tests full message parsing through the adapter.
func TestParseMessage_Valid(t *testing.T) {
	adapter := NewVLLMAdapter()

	// Build a valid msgpack payload with a BlockStored event
	blockStoredEvent := []any{
		"BlockStored",
		[]any{uint64(100), uint64(101)},
		uint64(99),
		[]uint32{1, 2, 3},
		16,
		nil,
		"gpu",
		nil,
		nil, // extra_keys
	}
	blockStoredPayload, err := msgpack.Marshal(blockStoredEvent)
	require.NoError(t, err)

	batch := []any{
		1234567890.0,
		[]any{blockStoredEvent},
		nil,
	}
	payload, err := msgpack.Marshal(batch)
	require.NoError(t, err)

	_ = blockStoredPayload // used indirectly via batch

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

// TestParseMessage_InvalidPayload tests error handling for invalid msgpack data.
func TestParseMessage_InvalidPayload(t *testing.T) {
	adapter := NewVLLMAdapter()

	msg := &kvevents.RawMessage{
		Topic:   "kv@pod-1@model",
		Payload: []byte{0xFF, 0xFF, 0xFF},
	}

	_, _, _, err := adapter.ParseMessage(msg)
	assert.Error(t, err)
}

// TestDecodeVLLMEvent_BlockStored tests decoding a valid BlockStored event without LoRA.
func TestDecodeVLLMEvent_BlockStored(t *testing.T) {
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
		nil, // extra_keys not present
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

// TestDecodeVLLMEvent_BlockStoredWithLora tests decoding a valid BlockStored event.
func TestDecodeVLLMEvent_BlockStoredWithLora(t *testing.T) {
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
		[]any{[]any{"uuid-A", "salt"}, nil}, // extra_keys
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

// TestDecodeVLLMEvent_BlockStoredMissingLoraName tests decoding with missing field.
func TestDecodeVLLMEvent_BlockStoredMissingLoraName(t *testing.T) {
	adapter := NewVLLMAdapter()

	vllmEvent := []any{
		"BlockStored",
		[]any{uint64(300), uint64(301)},
		uint64(299),
		[]uint32{7, 8, 9},
		64,
		123,
		"gpu",
	}

	rawBytes, err := msgpack.Marshal(vllmEvent)
	require.NoError(t, err)

	event, err := adapter.decodeVLLMEvent(rawBytes)
	assert.Error(t, err)
	assert.Nil(t, event)
}

// TestDecodeVLLMEvent_BlockStoredInvalidExtraKeys tests invalid extra_keys type.
func TestDecodeVLLMEvent_BlockStoredInvalidExtraKeys(t *testing.T) {
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
		[]any{"invalid_string"}, // Should be []any or nil, not string
	}

	rawBytes, err := msgpack.Marshal(vllmEvent)
	require.NoError(t, err)

	_, err = adapter.decodeVLLMEvent(rawBytes)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "extra_keys[0] has invalid type")
}

// TestDecodeVLLMEvent_BlockRemoved tests decoding a valid BlockRemoved event.
func TestDecodeVLLMEvent_BlockRemoved(t *testing.T) {
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

// TestDecodeVLLMEvent_AllBlocksCleared tests decoding a valid AllBlocksCleared event.
func TestDecodeVLLMEvent_AllBlocksCleared(t *testing.T) {
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

// TestDecodeVLLMEvent_UnknownTag tests error handling for unknown event tags.
func TestDecodeVLLMEvent_UnknownTag(t *testing.T) {
	adapter := NewVLLMAdapter()

	vllmEvent := []any{"UnknownEventType", "some", "data"}

	rawBytes, err := msgpack.Marshal(vllmEvent)
	require.NoError(t, err)

	event, err := adapter.decodeVLLMEvent(rawBytes)
	assert.Error(t, err)
	assert.Nil(t, event)
	assert.Contains(t, err.Error(), "unknown vLLM event tag")
}

// TestDecodeVLLMEvent_MalformedPayload tests error handling for malformed msgpack data.
func TestDecodeVLLMEvent_MalformedPayload(t *testing.T) {
	adapter := NewVLLMAdapter()

	rawBytes := []byte{0xFF, 0xFF, 0xFF}

	event, err := adapter.decodeVLLMEvent(rawBytes)
	assert.Error(t, err)
	assert.Nil(t, event)
}

// TestDecodeVLLMEvent_EmptyPayload tests error handling for empty event bytes.
func TestDecodeVLLMEvent_EmptyPayload(t *testing.T) {
	adapter := NewVLLMAdapter()

	rawBytes := []byte{}

	event, err := adapter.decodeVLLMEvent(rawBytes)
	assert.Error(t, err)
	assert.Nil(t, event)
}

// TestDecodeVLLMEvent_MissingTag tests error handling for events without a tag.
func TestDecodeVLLMEvent_MissingTag(t *testing.T) {
	adapter := NewVLLMAdapter()

	vllmEvent := []any{}

	rawBytes, err := msgpack.Marshal(vllmEvent)
	require.NoError(t, err)

	event, err := adapter.decodeVLLMEvent(rawBytes)
	assert.Error(t, err)
	assert.Nil(t, event)
	assert.Contains(t, err.Error(), "malformed tagged union")
}

// TestDecodeEventBatch_NestedArrayEvents tests that the batch decoder correctly handles
// events sent as nested msgpack arrays.
func TestDecodeEventBatch_NestedArrayEvents(t *testing.T) {
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
		nil, // extra_keys
	}

	batch := []any{
		1234567890.0,
		[]any{blockStoredEvent},
		nil,
	}

	payload, err := msgpack.Marshal(batch)
	require.NoError(t, err)

	// Decode the batch via ParseMessage
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

// TestGetHashAsUint64 tests hash format conversions.
func TestGetHashAsUint64(t *testing.T) {
	adapter := NewVLLMAdapter()

	t.Run("uint64", func(t *testing.T) {
		result, err := adapter.getHashAsUint64(uint64(42))
		require.NoError(t, err)
		assert.Equal(t, uint64(42), result)
	})

	t.Run("int64", func(t *testing.T) {
		result, err := adapter.getHashAsUint64(int64(42))
		require.NoError(t, err)
		assert.Equal(t, uint64(42), result)
	})

	t.Run("bytes_8", func(t *testing.T) {
		b := make([]byte, 8)
		binary.BigEndian.PutUint64(b, 12345)
		result, err := adapter.getHashAsUint64(b)
		require.NoError(t, err)
		assert.Equal(t, uint64(12345), result)
	})

	t.Run("bytes_empty", func(t *testing.T) {
		_, err := adapter.getHashAsUint64([]byte{})
		assert.Error(t, err)
	})

	t.Run("unsupported_type", func(t *testing.T) {
		_, err := adapter.getHashAsUint64("not a hash")
		assert.Error(t, err)
	})
}
