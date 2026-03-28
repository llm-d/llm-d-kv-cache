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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
)

// ---------------------------------------------------------------------------
// ParseRawExtraKeys
// ---------------------------------------------------------------------------

func TestParseRawExtraKeys_Nil(t *testing.T) {
	result, err := kvblock.ParseRawExtraKeys(nil)
	require.NoError(t, err)
	assert.Nil(t, result)
}

func TestParseRawExtraKeys_NilInnerEntries(t *testing.T) {
	raw := [][]any{nil, nil, nil}
	result, err := kvblock.ParseRawExtraKeys(raw)
	require.NoError(t, err)
	require.Len(t, result, 3)
	for _, r := range result {
		assert.Nil(t, r)
	}
}

func TestParseRawExtraKeys_ValidMMTuples(t *testing.T) {
	raw := [][]any{
		{[]any{"hash_A", int64(15)}}, // one MM entry
		{[]any{"hash_A", int64(-1)}}, // continuation
		nil,                          // text block
		{[]any{"hash_B", int64(2)}},  // different image
		{[]any{"hash_B", int64(-14)}, []any{"hash_C", int64(5)}}, // two overlapping images
	}

	result, err := kvblock.ParseRawExtraKeys(raw)
	require.NoError(t, err)
	require.Len(t, result, 5)

	require.NotNil(t, result[0])
	assert.Equal(t, "hash_A", result[0].MMHashes[0].Hash)
	assert.Equal(t, int64(15), result[0].MMHashes[0].Offset)

	require.NotNil(t, result[1])
	assert.Equal(t, int64(-1), result[1].MMHashes[0].Offset)

	assert.Nil(t, result[2])

	require.NotNil(t, result[3])
	assert.Equal(t, "hash_B", result[3].MMHashes[0].Hash)

	require.NotNil(t, result[4])
	require.Len(t, result[4].MMHashes, 2)
	assert.Equal(t, "hash_B", result[4].MMHashes[0].Hash)
	assert.Equal(t, "hash_C", result[4].MMHashes[1].Hash)
}

func TestParseRawExtraKeys_SkipsNonTupleEntries(t *testing.T) {
	// LoRA-style string pair entries should be skipped gracefully.
	raw := [][]any{
		{"uuid-A", "salt"}, // not a []any tuple — should be skipped
	}

	result, err := kvblock.ParseRawExtraKeys(raw)
	require.NoError(t, err)
	// No valid MM tuples found, so the block entry should be nil.
	assert.Nil(t, result[0])
}

func TestParseRawExtraKeys_VariousNumericTypes(t *testing.T) {
	// msgpack decodes integers as different Go types depending on magnitude.
	// All should be accepted as valid offsets.
	raw := [][]any{
		{[]any{"hash", int8(3)}},
		{[]any{"hash", int16(-17)}},
		{[]any{"hash", int32(100)}},
		{[]any{"hash", int64(-945)}},
		{[]any{"hash", uint8(15)}},
		{[]any{"hash", uint16(200)}},
		{[]any{"hash", uint32(300)}},
		{[]any{"hash", uint64(400)}},
		{[]any{"hash", int(42)}},
	}

	result, err := kvblock.ParseRawExtraKeys(raw)
	require.NoError(t, err)
	require.Len(t, result, 9)

	expected := []int64{3, -17, 100, -945, 15, 200, 300, 400, 42}
	for i, exp := range expected {
		require.NotNil(t, result[i], "block %d should have features", i)
		assert.Equal(t, exp, result[i].MMHashes[0].Offset, "block %d offset", i)
	}
}

func TestParseRawExtraKeys_UnsupportedOffsetType(t *testing.T) {
	raw := [][]any{
		{[]any{"hash", "not_a_number"}},
	}
	_, err := kvblock.ParseRawExtraKeys(raw)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported numeric type")
}

// ---------------------------------------------------------------------------
// ComputeBlockExtraFeatures
// ---------------------------------------------------------------------------

func TestComputeBlockExtraFeatures_NoOverlap(t *testing.T) {
	result := kvblock.ComputeBlockExtraFeatures(nil, nil, 16, 64)
	assert.Nil(t, result)
}

func TestComputeBlockExtraFeatures_SingleImage(t *testing.T) {
	// Image occupies tokens 0..47 (3 full blocks of size 16).
	mmHashes := map[string][]string{"image": {"hash_A"}}
	mmPlaceholders := map[string][]kvblock.PlaceholderRange{
		"image": {{Offset: 0, Length: 48}},
	}

	result := kvblock.ComputeBlockExtraFeatures(mmHashes, mmPlaceholders, 16, 64)
	require.Len(t, result, 4) // 64/16 = 4 blocks

	// Blocks 0-2 should have the image hash.
	require.NotNil(t, result[0])
	assert.Equal(t, "hash_A", result[0].MMHashes[0].Hash)
	assert.Equal(t, int64(0), result[0].MMHashes[0].Offset) // 0 - 0*16

	require.NotNil(t, result[1])
	assert.Equal(t, int64(-16), result[1].MMHashes[0].Offset) // 0 - 1*16

	require.NotNil(t, result[2])
	assert.Equal(t, int64(-32), result[2].MMHashes[0].Offset) // 0 - 2*16

	// Block 3 has no image overlap.
	assert.Nil(t, result[3])
}

func TestComputeBlockExtraFeatures_TextOnlyBlocksBetweenImages(t *testing.T) {
	// Two images with a gap: image 1 tokens [0, 32), text tokens [32, 48), image 2 tokens [48, 80).
	mmHashes := map[string][]string{"image": {"hashA", "hashB"}}
	mmPlaceholders := map[string][]kvblock.PlaceholderRange{
		"image": {{Offset: 0, Length: 32}, {Offset: 48, Length: 32}},
	}

	result := kvblock.ComputeBlockExtraFeatures(mmHashes, mmPlaceholders, 16, 80)
	require.Len(t, result, 5) // 80/16 = 5 blocks

	// Blocks 0,1: image A
	require.NotNil(t, result[0])
	assert.Equal(t, "hashA", result[0].MMHashes[0].Hash)
	require.NotNil(t, result[1])
	assert.Equal(t, "hashA", result[1].MMHashes[0].Hash)

	// Block 2: text only [32, 48) — no overlap with either image.
	assert.Nil(t, result[2])

	// Blocks 3,4: image B
	require.NotNil(t, result[3])
	assert.Equal(t, "hashB", result[3].MMHashes[0].Hash)
	require.NotNil(t, result[4])
	assert.Equal(t, "hashB", result[4].MMHashes[0].Hash)
}

// ---------------------------------------------------------------------------
// MM features affect block hashes — distinguishability
// ---------------------------------------------------------------------------

func TestMMFeatures_DifferentImagesProduceDifferentHashes(t *testing.T) {
	config := &kvblock.TokenProcessorConfig{BlockSize: 16, HashSeed: "test"}
	processor, err := kvblock.NewChunkedTokenDatabase(config)
	require.NoError(t, err)

	tokens := make([]uint32, 16) // one block
	for i := range tokens {
		tokens[i] = uint32(i + 1) // #nosec G115 -- test data, i < 64
	}

	// Same tokens, different image hashes → different block hashes.
	featA := []*kvblock.BlockExtraFeatures{
		{MMHashes: []kvblock.MMHash{{Hash: "image_hash_A", Offset: 0}}},
	}
	featB := []*kvblock.BlockExtraFeatures{
		{MMHashes: []kvblock.MMHash{{Hash: "image_hash_B", Offset: 0}}},
	}

	keysA, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", featA)
	require.NoError(t, err)
	keysB, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", featB)
	require.NoError(t, err)

	assert.NotEqual(t, keysA[0], keysB[0],
		"blocks with different image hashes must produce different block keys")
}

func TestMMFeatures_SameImageDifferentOffsetsProduceDifferentHashes(t *testing.T) {
	config := &kvblock.TokenProcessorConfig{BlockSize: 16, HashSeed: "test"}
	processor, err := kvblock.NewChunkedTokenDatabase(config)
	require.NoError(t, err)

	tokens := make([]uint32, 16)
	for i := range tokens {
		tokens[i] = uint32(i + 1) // #nosec G115 -- test data, i < 64
	}

	offsets := []int64{15, -1, -17, -33}
	hashes := make([]kvblock.BlockHash, len(offsets))
	for i, off := range offsets {
		feat := []*kvblock.BlockExtraFeatures{
			{MMHashes: []kvblock.MMHash{{Hash: "same_hash", Offset: off}}},
		}
		keys, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", feat)
		require.NoError(t, err)
		hashes[i] = keys[0]
	}

	// All offsets should produce distinct hashes.
	seen := make(map[kvblock.BlockHash]int64)
	for i, h := range hashes {
		if prev, ok := seen[h]; ok {
			t.Errorf("offsets %d and %d produced the same hash", prev, offsets[i])
		}
		seen[h] = offsets[i]
	}
}

func TestMMFeatures_NilFeaturesSameAsTextOnly(t *testing.T) {
	config := &kvblock.TokenProcessorConfig{BlockSize: 16, HashSeed: "test"}
	processor, err := kvblock.NewChunkedTokenDatabase(config)
	require.NoError(t, err)

	tokens := make([]uint32, 16)
	for i := range tokens {
		tokens[i] = uint32(i + 1) // #nosec G115 -- test data, i < 64
	}

	// nil extraFeatures = text-only
	keysNil, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", nil)
	require.NoError(t, err)

	// Explicit nil entry per block = also text-only
	keysExplicitNil, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model",
		[]*kvblock.BlockExtraFeatures{nil})
	require.NoError(t, err)

	assert.Equal(t, keysNil[0], keysExplicitNil[0],
		"nil extraFeatures and explicit nil-per-block must produce the same hash")
}

func TestMMFeatures_OnlyAffectOverlappingBlocks(t *testing.T) {
	config := &kvblock.TokenProcessorConfig{BlockSize: 16, HashSeed: "test"}
	processor, err := kvblock.NewChunkedTokenDatabase(config)
	require.NoError(t, err)

	// 4 blocks of tokens
	tokens := make([]uint32, 64)
	for i := range tokens {
		tokens[i] = uint32(i + 1) // #nosec G115 -- test data, i < 64
	}

	// Text-only baseline
	keysTextOnly, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", nil)
	require.NoError(t, err)
	require.Len(t, keysTextOnly, 4)

	// Image only in block 2 (index 2)
	features := []*kvblock.BlockExtraFeatures{
		nil, // block 0: text
		nil, // block 1: text
		{MMHashes: []kvblock.MMHash{{Hash: "image_X", Offset: 5}}}, // block 2: image
		nil, // block 3: text
	}
	keysWithImage, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", features)
	require.NoError(t, err)
	require.Len(t, keysWithImage, 4)

	// Blocks 0 and 1 should be identical (prefix chain before image block).
	assert.Equal(t, keysTextOnly[0], keysWithImage[0], "block 0 should be unaffected by image in block 2")
	assert.Equal(t, keysTextOnly[1], keysWithImage[1], "block 1 should be unaffected by image in block 2")

	// Block 2 should differ (image taints the hash).
	assert.NotEqual(t, keysTextOnly[2], keysWithImage[2], "block 2 should be affected by image")

	// Block 3 should also differ because the prefix chain changed at block 2.
	assert.NotEqual(t, keysTextOnly[3], keysWithImage[3],
		"block 3 hash should change because it chains from the tainted block 2")
}

func TestMMFeatures_MismatchedLengthReturnsError(t *testing.T) {
	config := &kvblock.TokenProcessorConfig{BlockSize: 16, HashSeed: "test"}
	processor, err := kvblock.NewChunkedTokenDatabase(config)
	require.NoError(t, err)

	tokens := make([]uint32, 32)                   // 2 chunks
	features := []*kvblock.BlockExtraFeatures{nil} // 1 entry — mismatch

	_, err = processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", features)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "does not match token chunk count")
}
