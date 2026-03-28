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

package kvblock_test

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents/engineadapter"
)

// TestMMVLLMCapture_IngestionMatchesRequestPath cross-validates the real vLLM-captured
// msgpack file against ComputeBlockExtraFeatures.
//
// The captured file (block_stored_example.msgpack) was recorded from a live
// Qwen/Qwen2-VL-2B-Instruct inference with a two-image prompt. It contains
// 117 blocks (block_size=16, 1872 tokens) with two distinct image hashes
// and per-block extra_keys encoding (mm_hash, block_relative_offset) tuples.
//
// This test proves that:
//  1. ParseRawExtraKeys correctly parses real vLLM extra_keys wire format
//  2. ComputeBlockExtraFeatures from placeholder ranges produces the exact
//     same per-block features that vLLM sent in the BlockStored event
//  3. Both paths produce identical block keys — the ingestion and request
//     paths agree on real multimodal data
func TestMMVLLMCapture_IngestionMatchesRequestPath(t *testing.T) {
	// Load the real vLLM capture.
	payload, err := os.ReadFile("../../../examples/testdata/block_stored_example.msgpack")
	require.NoError(t, err, "captured msgpack file must exist")

	// Parse through the VLLMAdapter.
	adapter := engineadapter.NewVLLMAdapter()
	_, _, batch, err := adapter.ParseMessage(&kvevents.RawMessage{
		Topic:   "kv@pod-1@Qwen/Qwen2-VL-2B-Instruct",
		Payload: payload,
	})
	require.NoError(t, err)
	require.Len(t, batch.Events, 1)

	ev, ok := batch.Events[0].(*kvevents.BlockStoredEvent)
	require.True(t, ok)
	assert.Len(t, ev.BlockHashes, 117)
	assert.Len(t, ev.Tokens, 1872)
	assert.Equal(t, "GPU", ev.DeviceTier)
	assert.Len(t, ev.ExtraKeys, 117)

	// INGESTION PATH: parse vLLM's extra_keys.
	ingested, err := kvblock.ParseRawExtraKeys(ev.ExtraKeys)
	require.NoError(t, err)

	// Verify the structure matches known layout:
	// Image 1 (6ab3a7d...): blocks 0-60, first offset=15
	// Block 61: nil (text)
	// Image 2 (e950785...): blocks 62-115, first offset=2
	// Block 116: nil (text)
	const (
		img1Hash = "6ab3a7d0570817f1a4e9adaeda325c07c2466b252279a633ee2995cdba59ab25"
		img2Hash = "e950785918bdef0f88ec349d3f65a2ed0b1d448c854333ea1e71bfedce1fe252"
	)

	require.NotNil(t, ingested[0])
	assert.Equal(t, img1Hash, ingested[0].MMHashes[0].Hash)
	assert.Equal(t, int64(15), ingested[0].MMHashes[0].Offset)

	require.NotNil(t, ingested[60])
	assert.Equal(t, img1Hash, ingested[60].MMHashes[0].Hash)
	assert.Equal(t, int64(15-60*16), ingested[60].MMHashes[0].Offset)

	assert.Nil(t, ingested[61], "block 61 should be text-only")

	require.NotNil(t, ingested[62])
	assert.Equal(t, img2Hash, ingested[62].MMHashes[0].Hash)
	assert.Equal(t, int64(2), ingested[62].MMHashes[0].Offset)

	assert.Nil(t, ingested[116], "block 116 should be text-only")

	// REQUEST PATH: reconstruct from placeholder ranges.
	// Derived from the capture: image 1 starts at token 15, covers blocks 0-60 (ends at 976).
	// Image 2 starts at token 994 (offset 2 in block 62 = 2 + 62*16), covers blocks 62-115 (ends at 1856).
	mmHashes := map[string][]string{
		"image": {img1Hash, img2Hash},
	}
	mmPlaceholders := map[string][]kvblock.PlaceholderRange{
		"image": {
			{Offset: 15, Length: 961},  // 976 - 15
			{Offset: 994, Length: 862}, // 1856 - 994
		},
	}

	computed := kvblock.ComputeBlockExtraFeatures(mmHashes, mmPlaceholders, 16, 1872)
	require.Len(t, computed, 117)

	// Compare every block: ingested (from vLLM wire) vs computed (from placeholder ranges).
	for i := range ingested {
		if ingested[i] == nil {
			assert.Nil(t, computed[i], "block %d: vLLM sent nil, computed should be nil", i)
			continue
		}
		require.NotNil(t, computed[i], "block %d: vLLM sent features, computed should too", i)
		require.Len(t, computed[i].MMHashes, len(ingested[i].MMHashes),
			"block %d: entry count mismatch", i)

		for j := range ingested[i].MMHashes {
			assert.Equal(t, ingested[i].MMHashes[j].Hash, computed[i].MMHashes[j].Hash,
				"block %d entry %d: hash mismatch", i, j)
			assert.Equal(t, ingested[i].MMHashes[j].Offset, computed[i].MMHashes[j].Offset,
				"block %d entry %d: offset mismatch", i, j)
		}
	}

	// Verify block keys match between ingestion and request paths.
	proc, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{BlockSize: 16})
	require.NoError(t, err)

	keysIngested, err := proc.TokensToKVBlockKeys(
		kvblock.EmptyBlockHash, ev.Tokens, "Qwen/Qwen2-VL-2B-Instruct", ingested)
	require.NoError(t, err)

	keysComputed, err := proc.TokensToKVBlockKeys(
		kvblock.EmptyBlockHash, ev.Tokens, "Qwen/Qwen2-VL-2B-Instruct", computed)
	require.NoError(t, err)

	for i := range keysIngested {
		assert.Equal(t, keysIngested[i], keysComputed[i],
			"block %d: ingestion key != request key", i)
	}
}
