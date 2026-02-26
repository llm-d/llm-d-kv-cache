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

package kvevents_test

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/vmihailenco/msgpack/v5"

	. "github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
)

// Helper function to create BlockStored raw msgpack message.
func createBlockStoredRaw(t *testing.T, fields []any) msgpack.RawMessage {
	t.Helper()
	data, err := msgpack.Marshal(fields)
	if err != nil {
		t.Fatalf("Failed to marshal fields: %v", err)
	}
	return msgpack.RawMessage(data)
}

func TestBlockStoredMissingLoraName(t *testing.T) {
	rawMsg := createBlockStoredRaw(t, []any{
		BlockStoredEventTag,               // Event tag
		[]any{uint64(1001), uint64(1002)}, // BlockHashes
		nil,                               // ParentBlockHash
		[]uint32{1, 2, 3},                 // TokenIds
		256,                               // BlockSize
		42,                                // LoraID
		"GPU",                             // Medium
		// LoraName is missing
	})

	_, err := UnmarshalKVEvent(rawMsg)

	// Expect error due to missing LoraName
	require.Error(t, err)
}

func TestBlockStoredAllFieldsPresent(t *testing.T) {
	rawMsg := createBlockStoredRaw(t, []any{
		BlockStoredEventTag,               // Event tag
		[]any{uint64(1001), uint64(1002)}, // BlockHashes
		nil,                               // ParentBlockHash
		[]uint32{1, 2, 3},                 // TokenIds
		256,                               // BlockSize
		42,                                // LoraID
		"GPU",                             // Medium
		"test-lora",                       // LoraName
	})

	event, err := UnmarshalKVEvent(rawMsg)

	require.NoError(t, err, "Expected no error during unmarshaling")
	require.NotNil(t, event, "Expected event to be non-nil")

	blockStored, ok := event.(BlockStored)
	require.True(t, ok, "Expected event to be of type BlockStored")

	if blockStored.Medium == nil || *blockStored.Medium != "GPU" {
		t.Errorf("Expected Medium to be 'GPU', got %v", blockStored.Medium)
	}
	require.NotNil(t, blockStored.Medium, "Expected Medium to be non-nil")
	require.Equal(t, "GPU", *blockStored.Medium, "Expected Medium to be 'GPU'")

	require.NotNil(t, blockStored.LoraName, "Expected LoraName to be non-nil")
	require.Equal(t, "test-lora", *blockStored.LoraName, "Expected LoraName to be 'test-lora'")
}

func TestUnmarshalKVEventErrors(t *testing.T) {
	// Test unknown event tag
	rawMsg := createBlockStoredRaw(t, []any{
		BlockStoredEventTag,               // Event tag
		[]any{uint64(1001), uint64(1002)}, // BlockHashes
		nil,                               // ParentBlockHash
		[]uint32{1, 2, 3},                 // TokenIds
	})

	var err error
	_, err = UnmarshalKVEvent(rawMsg)
	require.Error(t, err, "Expected error for incomplete BlockStored event")

	// Test malformed union (empty array)
	emptyRawMsg := createBlockStoredRaw(t, []any{})
	_, err = UnmarshalKVEvent(emptyRawMsg)
	require.Error(t, err, "Expected error for malformed tagged union")
}
