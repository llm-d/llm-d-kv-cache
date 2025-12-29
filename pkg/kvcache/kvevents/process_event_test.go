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

	"github.com/vmihailenco/msgpack/v5"

	. "github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvevents"
)

// Helper function to create BlockStored raw msgpack message
func createBlockStoredRaw(t *testing.T, fields []any) msgpack.RawMessage {
	data, err := msgpack.Marshal(fields)
	if err != nil {
		t.Fatalf("Failed to marshal fields: %v", err)
	}
	return msgpack.RawMessage(data)
}

func TestBlockStoredMissingMediumAndLoraName(t *testing.T) {
	rawMsg := createBlockStoredRaw(t, []any{
		BlockStoredEventTag,               // Event tag
		[]any{uint64(1001), uint64(1002)}, // BlockHashes
		nil,                               // ParentBlockHash
		[]uint32{1, 2, 3},                 // TokenIds
		256,                               // BlockSize
		42,                                // LoraID
		// Medium and LoraName are missing
	})

	event, err := UnmarshalKVEvent(rawMsg)
	if err != nil {
		t.Fatalf("Failed to process BlockStored event: %v", err)
	}

	if event == nil {
		t.Error("Expected event to be non-nil")
	}

	blockStored, ok := event.(BlockStored)
	if !ok {
		t.Fatalf("Expected BlockStored event, got %T", event)
	}

	if blockStored.Medium != nil {
		t.Errorf("Expected Medium to be nil, got %v", *blockStored.Medium)
	}
	if blockStored.LoraName != nil {
		t.Errorf("Expected LoraName to be nil, got %v", *blockStored.LoraName)
	}
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

	event, err := UnmarshalKVEvent(rawMsg)
	if err != nil {
		t.Fatalf("Failed to process BlockStored event: %v", err)
	}

	if event == nil {
		t.Error("Expected event to be non-nil")
	}

	blockStored, ok := event.(BlockStored)
	if !ok {
		t.Fatalf("Expected BlockStored event, got %T", event)
	}

	if blockStored.Medium == nil || *blockStored.Medium != "GPU" {
		t.Errorf("Expected Medium to be 'GPU', got %v", blockStored.Medium)
	}
	if blockStored.LoraName != nil {
		t.Errorf("Expected LoraName to be nil, got %v", *blockStored.LoraName)
	}
}

func TestBlockStoredAllFieldsPresent(t *testing.T) {
	rawMsg := createBlockStoredRaw(t, []any{
		BlockStoredEventTag,               // Event tag
		[]any{uint64(1001), uint64(1002)}, // BlockHashes
		nil,                               // ParentBlockHash
		[]uint32{1, 2, 3},                 // TokenIds
		256,                               // BlockSize
		42,                                // LoraID
		"gpu",                             // Medium
		"test-lora",                       // LoraName
	})

	event, err := UnmarshalKVEvent(rawMsg)
	if err != nil {
		t.Fatalf("Failed to process BlockStored event: %v", err)
	}

	if event == nil {
		t.Error("Expected event to be non-nil")
	}

	blockStored, ok := event.(BlockStored)
	if !ok {
		t.Fatalf("Expected BlockStored event, got %T", event)
	}

	if blockStored.Medium == nil || *blockStored.Medium != "gpu" {
		t.Errorf("Expected Medium to be 'gpu', got %v", blockStored.Medium)
	}
	if blockStored.LoraName == nil || *blockStored.LoraName != "test-lora" {
		t.Errorf("Expected LoraName to be 'test-lora', got %v", blockStored.LoraName)
	}
}

func TestUnmarshalKVEventErrors(t *testing.T) {
	// Test invalid msgpack
	_, err := UnmarshalKVEvent(msgpack.RawMessage([]byte{0x01, 0x02, 0x03}))
	if err == nil {
		t.Error("Expected error for invalid msgpack")
	}

	// Test unknown event tag
	rawMsg := createBlockStoredRaw(t, []any{
		"UnknownEvent",
		[]any{uint64(1001)},
	})
	_, err = UnmarshalKVEvent(rawMsg)
	if err == nil {
		t.Error("Expected error for unknown event tag")
	}

	// Test malformed union (empty array)
	emptyRawMsg := createBlockStoredRaw(t, []any{})
	_, err = UnmarshalKVEvent(emptyRawMsg)
	if err == nil {
		t.Error("Expected error for malformed union")
	}
}
