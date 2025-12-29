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

package kvevents

import (
	"testing"

	"github.com/vmihailenco/msgpack/v5"
)

// Helper function to create BlockStored raw msgpack message
func createBlockStoredRaw(t *testing.T, fields []any) msgpack.RawMessage {
	data, err := msgpack.Marshal(fields)
	if err != nil {
		t.Fatalf("Failed to marshal fields: %v", err)
	}
	return msgpack.RawMessage(data)
}

func TestProcessBlockStoredKVEvent(t *testing.T) {
	// Create a raw msgpack.RawMessage for BlockStored event with missing fields

	// Test case 1: Missing both Medium and LoraName
	missingBothFields := createBlockStoredRaw(t, []any{
		[]any{uint64(1001), uint64(1002)}, // BlockHashes
		nil,                               // ParentBlockHash
		[]uint32{1, 2, 3},                 // TokenIds
		256,                               // BlockSize
		42,                                // LoraID
		// Medium and LoraName are missing
	})

	// Test case 2: Missing only LoraName
	missingLoraName := createBlockStoredRaw(t, []any{
		[]any{uint64(1001), uint64(1002)}, // BlockHashes
		nil,                               // ParentBlockHash
		[]uint32{1, 2, 3},                 // TokenIds
		256,                               // BlockSize
		42,                                // LoraID
		"cpu",                             // Medium
		// LoraName is missing
	})

	// Test case 3: All fields present
	allFieldsPresent := createBlockStoredRaw(t, []any{
		[]any{uint64(1001), uint64(1002)}, // BlockHashes
		nil,                               // ParentBlockHash
		[]uint32{1, 2, 3},                 // TokenIds
		256,                               // BlockSize
		42,                                // LoraID
		"gpu",                             // Medium
		"test-lora",                       // LoraName
	})

	// Use the raw messages for testing
	testCases := []struct {
		name   string
		rawMsg msgpack.RawMessage
	}{
		{"missing_medium_and_lora_name", missingBothFields},
		{"missing_lora_name", missingLoraName},
		{"all_fields_present", allFieldsPresent},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := unmarshalKVEvent(tc.rawMsg)
			if err != nil {
				t.Fatalf("Failed to process BlockStored event: %v", err)
			}

			switch tc.name {
			case "missing_medium_and_lora_name":
				if len(tc.rawMsg) != 5 {
					t.Errorf("Expected 5 fields, got %d", len(tc.rawMsg))
				}
			case "missing_lora_name":
				if len(tc.rawMsg) != 6 {
					t.Errorf("Expected 6 fields, got %d", len(tc.rawMsg))
				}
			case "all_fields_present":
				if len(tc.rawMsg) != 7 {
					t.Errorf("Expected 7 fields, got %d", len(tc.rawMsg))
				}
			}
		})
	}
}
