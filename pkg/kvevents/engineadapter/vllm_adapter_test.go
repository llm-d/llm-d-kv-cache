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
	"encoding/binary"
	"testing"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents/events"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/vmihailenco/msgpack/v5"
)

// TestParseVLLMMessage_Valid tests parsing a valid 3-part vLLM message.
func TestParseVLLMMessage_Valid(t *testing.T) {
	topic := []byte("pod-123@llama-2-7b")
	sequence := make([]byte, 8)
	binary.BigEndian.PutUint64(sequence, 42)
	payload := []byte("test payload")

	parts := [][]byte{topic, sequence, payload}

	msg, err := parseVLLMMessage(parts)
	require.NoError(t, err)
	assert.Equal(t, "pod-123@llama-2-7b", msg.Topic)
	assert.Equal(t, uint64(42), msg.Sequence)
	assert.Equal(t, payload, msg.Payload)
}

// TestParseVLLMMessage_InvalidParts tests error handling for messages with invalid parts number.
func TestParseVLLMMessage_TooFewParts(t *testing.T) {
	parts := [][]byte{
		[]byte("topic"),
		[]byte("sequence"),
	}

	msg, err := parseVLLMMessage(parts)
	assert.Error(t, err)
	assert.Nil(t, msg)
	assert.Contains(t, err.Error(), "expected 3 message parts")
}

// TestParseVLLMMessage_TooManyParts tests error handling for messages with more than 3 parts.
func TestParseVLLMMessage_TooManyParts(t *testing.T) {
	sequence := make([]byte, 8)
	binary.BigEndian.PutUint64(sequence, 1)

	parts := [][]byte{
		[]byte("topic"),
		sequence,
		[]byte("payload"),
		[]byte("extra"),
	}

	msg, err := parseVLLMMessage(parts)
	assert.Error(t, err)
	assert.Nil(t, msg)
	assert.Contains(t, err.Error(), "expected 3 message parts")
}

// TestParseVLLMMessage_InvalidSequenceLength tests error handling for invalid sequence byte length.
// vLLM sends sequence numbers as 8-byte big-endian uint64. This test verifies that parseVLLMMessage
// correctly rejects messages with sequence byte lengths that are not exactly 8 bytes (empty, too short, or too long).
func TestParseVLLMMessage_InvalidSequenceLength(t *testing.T) {
	testCases := []struct {
		name           string
		sequenceLength int
	}{
		{"empty", 0},
		{"too short", 4},
		{"too long", 16},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			parts := [][]byte{
				[]byte("topic"),
				make([]byte, tc.sequenceLength),
				[]byte("payload"),
			}

			msg, err := parseVLLMMessage(parts)
			assert.Error(t, err)
			assert.Nil(t, msg)
			assert.Contains(t, err.Error(), "invalid sequence bytes length")
		})
	}
}

// TestParseVLLMMessage_MaxSequence tests parsing with maximum sequence number.
func TestParseVLLMMessage_MaxSequence(t *testing.T) {
	sequence := make([]byte, 8)
	binary.BigEndian.PutUint64(sequence, ^uint64(0)) // Max uint64

	parts := [][]byte{
		[]byte("topic"),
		sequence,
		[]byte("payload"),
	}

	msg, err := parseVLLMMessage(parts)
	require.NoError(t, err)
	assert.Equal(t, ^uint64(0), msg.Sequence)
}

// TestDecodeVLLMEvent_BlockStored tests decoding a valid BlockStored event without LoRA.
func TestDecodeVLLMEvent_BlockStored(t *testing.T) {
	adapter, err := NewVLLMAdapter()
	require.NoError(t, err)
	defer adapter.Close()

	// Create a BlockStored event without LoRA
	// Note: With array encoding, all fields must be present (nil for unused LoRA fields)
	vllmEvent := []any{
		"BlockStored",                   // Tag
		[]any{uint64(100), uint64(101)}, // BlockHashes
		uint64(99),                      // ParentBlockHash
		[]uint32{1, 2, 3},               // TokenIds
		16,                              // BlockSize
		nil,                             // LoraID (nil when not using LoRA)
		"gpu",                           // Medium
		nil,                             // LoraName (nil when not using LoRA)
	}

	rawBytes, err := msgpack.Marshal(vllmEvent)
	require.NoError(t, err)

	event, err := adapter.decodeVLLMEvent(rawBytes)
	require.NoError(t, err)
	require.NotNil(t, event)

	blockStored, ok := event.(*events.BlockStoredEvent)
	require.True(t, ok, "expected BlockStoredEvent")
	assert.Equal(t, []uint64{100, 101}, blockStored.BlockHashes)
	assert.Equal(t, uint64(99), blockStored.ParentHash)
	assert.Equal(t, []uint32{1, 2, 3}, blockStored.Tokens)
	assert.Equal(t, "gpu", blockStored.DeviceTier)
	assert.Nil(t, blockStored.LoraID)
	assert.Nil(t, blockStored.LoraName)
}

// TestDecodeVLLMEvent_BlockStoredWithLora tests decoding a valid BlockStored event
func TestDecodeVLLMEvent_BlockStoredWithLora(t *testing.T) {
	adapter, err := NewVLLMAdapter()
	require.NoError(t, err)
	defer adapter.Close()

	// Create a BlockStored event with LoRA fields
	vllmEvent := []any{
		"BlockStored",                   // Tag (not part of struct)
		[]any{uint64(200), uint64(201)}, // BlockHashes
		uint64(199),                     // ParentBlockHash
		[]uint32{4, 5, 6},               // TokenIds
		32,                              // BlockSize
		42,                              // LoraID
		"gpu",                           // Medium
		"test-lora",                     // LoraName
	}

	rawBytes, err := msgpack.Marshal(vllmEvent)
	require.NoError(t, err)

	event, err := adapter.decodeVLLMEvent(rawBytes)
	require.NoError(t, err)
	require.NotNil(t, event)

	blockStored, ok := event.(*events.BlockStoredEvent)
	require.True(t, ok, "expected BlockStoredEvent")
	assert.Equal(t, []uint64{200, 201}, blockStored.BlockHashes)
	assert.Equal(t, uint64(199), blockStored.ParentHash)
	assert.Equal(t, []uint32{4, 5, 6}, blockStored.Tokens)
	assert.Equal(t, "gpu", blockStored.DeviceTier)
	require.NotNil(t, blockStored.LoraID)
	assert.Equal(t, 42, *blockStored.LoraID)
	require.NotNil(t, blockStored.LoraName)
	assert.Equal(t, "test-lora", *blockStored.LoraName)
}

// TestDecodeVLLMEvent_BlockStoredMissingLoraName tests decoding with LoraID but missing LoraName.
func TestDecodeVLLMEvent_BlockStoredMissingLoraName(t *testing.T) {
	adapter, err := NewVLLMAdapter()
	require.NoError(t, err)
	defer adapter.Close()

	// Create a BlockStored event with LoraID but no LoraName (6 fields - invalid)
	// vLLM should send either all LoRA fields or none
	vllmEvent := []any{
		"BlockStored",                   // Tag (not part of struct)
		[]any{uint64(300), uint64(301)}, // BlockHashes
		uint64(299),                     // ParentBlockHash
		[]uint32{7, 8, 9},               // TokenIds
		64,                              // BlockSize
		123,                             // LoraID
		"gpu",                           // Medium
	}

	rawBytes, err := msgpack.Marshal(vllmEvent)
	require.NoError(t, err)

	// This should fail during msgpack decoding because the struct expects 8 fields
	event, err := adapter.decodeVLLMEvent(rawBytes)
	assert.Error(t, err)
	assert.Nil(t, event)
}

// TestDecodeVLLMEvent_BlockRemoved tests decoding a valid BlockRemoved event.
func TestDecodeVLLMEvent_BlockRemoved(t *testing.T) {
	adapter, err := NewVLLMAdapter()
	require.NoError(t, err)
	defer adapter.Close()

	// Create a BlockRemoved event in vLLM format
	// The struct has 2 fields: BlockHashes, Medium
	medium := "cpu"
	vllmEvent := []any{
		"BlockRemoved", // Tag (not part of struct)
		[]any{uint64(200), uint64(201), uint64(202)}, // BlockHashes
		&medium, // Medium (optional)
	}

	rawBytes, err := msgpack.Marshal(vllmEvent)
	require.NoError(t, err)

	event, err := adapter.decodeVLLMEvent(rawBytes)
	require.NoError(t, err)
	require.NotNil(t, event)

	blockRemoved, ok := event.(*events.BlockRemovedEvent)
	require.True(t, ok, "expected BlockRemovedEvent")
	assert.Equal(t, []uint64{200, 201, 202}, blockRemoved.BlockHashes)
	assert.Equal(t, "cpu", blockRemoved.DeviceTier)
}

// TestDecodeVLLMEvent_AllBlocksCleared tests decoding a valid AllBlocksCleared event.
func TestDecodeVLLMEvent_AllBlocksCleared(t *testing.T) {
	adapter, err := NewVLLMAdapter()
	require.NoError(t, err)
	defer adapter.Close()

	// Create an AllBlocksCleared event in vLLM format
	vllmEvent := []any{"AllBlocksCleared"}

	rawBytes, err := msgpack.Marshal(vllmEvent)
	require.NoError(t, err)

	event, err := adapter.decodeVLLMEvent(rawBytes)
	require.NoError(t, err)
	require.NotNil(t, event)

	_, ok := event.(*events.AllBlocksClearedEvent)
	require.True(t, ok, "expected AllBlocksClearedEvent")
}

// TestDecodeVLLMEvent_UnknownTag tests error handling for unknown event tags.
func TestDecodeVLLMEvent_UnknownTag(t *testing.T) {
	adapter, err := NewVLLMAdapter()
	require.NoError(t, err)
	defer adapter.Close()

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
	adapter, err := NewVLLMAdapter()
	require.NoError(t, err)
	defer adapter.Close()

	// Invalid msgpack data
	rawBytes := []byte{0xFF, 0xFF, 0xFF}

	event, err := adapter.decodeVLLMEvent(rawBytes)
	assert.Error(t, err)
	assert.Nil(t, event)
}

// TestDecodeVLLMEvent_EmptyPayload tests error handling for empty event bytes.
func TestDecodeVLLMEvent_EmptyPayload(t *testing.T) {
	adapter, err := NewVLLMAdapter()
	require.NoError(t, err)
	defer adapter.Close()

	rawBytes := []byte{}

	event, err := adapter.decodeVLLMEvent(rawBytes)
	assert.Error(t, err)
	assert.Nil(t, event)
}

// TestDecodeVLLMEvent_MissingTag tests error handling for events without a tag.
func TestDecodeVLLMEvent_MissingTag(t *testing.T) {
	adapter, err := NewVLLMAdapter()
	require.NoError(t, err)
	defer adapter.Close()

	// Empty array - no tag
	vllmEvent := []any{}

	rawBytes, err := msgpack.Marshal(vllmEvent)
	require.NoError(t, err)

	event, err := adapter.decodeVLLMEvent(rawBytes)
	assert.Error(t, err)
	assert.Nil(t, event)
	assert.Contains(t, err.Error(), "malformed tagged union")
}
