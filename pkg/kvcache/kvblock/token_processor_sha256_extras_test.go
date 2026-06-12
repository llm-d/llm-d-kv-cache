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

// Tests for SHA256-CBOR hashing with extra features support (LoRA, MM, salt, embeds).

package kvblock_test

import (
	"crypto/sha256"
	"testing"

	"github.com/fxamacker/cbor/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
)

// ---------------------------------------------------------------------------
// Test Configuration Helpers
// ---------------------------------------------------------------------------

// sha256ConfigUnified returns a TokenProcessorConfig using the unified fields
// (HashSeed, BlockSizeTokens) for SHA256-CBOR mode.
func sha256ConfigUnified(seed string, blockSize int) *kvblock.TokenProcessorConfig {
	return &kvblock.TokenProcessorConfig{
		HashAlgorithm:   kvblock.HashAlgorithmSHA256CBOR,
		HashSeed:        seed,
		BlockSizeTokens: blockSize,
	}
}

// ---------------------------------------------------------------------------
// BlockExtraFeatures.CborExtras() Tests
// ---------------------------------------------------------------------------

func TestBlockExtraFeatures_cborExtras_Nil(t *testing.T) {
	var ef *kvblock.BlockExtraFeatures
	result := ef.CborExtras()
	assert.Nil(t, result, "nil receiver should return nil")
}

func TestBlockExtraFeatures_cborExtras_EmptyStruct(t *testing.T) {
	ef := &kvblock.BlockExtraFeatures{}
	result := ef.CborExtras()
	assert.Nil(t, result, "empty struct should return nil")
}

func TestBlockExtraFeatures_cborExtras_RawExtrasTakesPrecedence(t *testing.T) {
	ef := &kvblock.BlockExtraFeatures{
		RawExtras: []any{"raw1", "raw2"},
		LoraName:  "should-be-ignored",
		MMHashes:  []kvblock.MMHash{{Hash: "should-be-ignored"}},
	}
	result := ef.CborExtras()
	require.NotNil(t, result)

	arr, ok := result.([]any)
	require.True(t, ok)
	require.Len(t, arr, 2)
	assert.Equal(t, "raw1", arr[0])
	assert.Equal(t, "raw2", arr[1])
}

func TestBlockExtraFeatures_cborExtras_EmptyRawExtrasReturnsNil(t *testing.T) {
	ef := &kvblock.BlockExtraFeatures{
		RawExtras: []any{},
	}
	result := ef.CborExtras()
	assert.Nil(t, result, "empty RawExtras should return nil")
}

func TestBlockExtraFeatures_cborExtras_LoraOnly(t *testing.T) {
	ef := &kvblock.BlockExtraFeatures{
		LoraName: "my-lora-adapter",
	}
	result := ef.CborExtras()
	require.NotNil(t, result)

	arr, ok := result.([]any)
	require.True(t, ok)
	require.Len(t, arr, 1)
	assert.Equal(t, "my-lora-adapter", arr[0])
}

func TestBlockExtraFeatures_cborExtras_MMHashesOnly(t *testing.T) {
	ef := &kvblock.BlockExtraFeatures{
		MMHashes: []kvblock.MMHash{
			{Hash: "mm_hash_1"},
			{Hash: "mm_hash_2"},
		},
	}
	result := ef.CborExtras()
	require.NotNil(t, result)

	arr, ok := result.([]any)
	require.True(t, ok)
	require.Len(t, arr, 2)
	assert.Equal(t, "mm_hash_1", arr[0])
	assert.Equal(t, "mm_hash_2", arr[1])
}

func TestBlockExtraFeatures_cborExtras_CacheSaltOnly(t *testing.T) {
	ef := &kvblock.BlockExtraFeatures{
		CacheSalt: "debug-session-123",
	}
	result := ef.CborExtras()
	require.NotNil(t, result)

	arr, ok := result.([]any)
	require.True(t, ok)
	require.Len(t, arr, 1)
	assert.Equal(t, "debug-session-123", arr[0])
}

func TestBlockExtraFeatures_cborExtras_PromptEmbedsOnly(t *testing.T) {
	embedsHash := []byte{0x01, 0x02, 0x03, 0x04}
	ef := &kvblock.BlockExtraFeatures{
		PromptEmbedsHash: embedsHash,
	}
	result := ef.CborExtras()
	require.NotNil(t, result)

	arr, ok := result.([]any)
	require.True(t, ok)
	require.Len(t, arr, 1)
	assert.Equal(t, embedsHash, arr[0])
}

func TestBlockExtraFeatures_cborExtras_CorrectOrdering(t *testing.T) {
	// vLLM order: [LoRA?, MM..., Salt?, Embeds?]
	embedsHash := []byte{0xaa, 0xbb}
	ef := &kvblock.BlockExtraFeatures{
		LoraName:         "lora-adapter",
		MMHashes:         []kvblock.MMHash{{Hash: "mm1"}, {Hash: "mm2"}},
		CacheSalt:        "salt-xyz",
		PromptEmbedsHash: embedsHash,
	}
	result := ef.CborExtras()
	require.NotNil(t, result)

	arr, ok := result.([]any)
	require.True(t, ok)
	require.Len(t, arr, 5)

	assert.Equal(t, "lora-adapter", arr[0], "LoRA should be first")
	assert.Equal(t, "mm1", arr[1], "MM hash 1 should be second")
	assert.Equal(t, "mm2", arr[2], "MM hash 2 should be third")
	assert.Equal(t, "salt-xyz", arr[3], "Salt should be fourth")
	assert.Equal(t, embedsHash, arr[4], "Embeds should be last")
}

// ---------------------------------------------------------------------------
// SHA256 with Extra Features Integration Tests
// ---------------------------------------------------------------------------

func TestSHA256_ExtraFeatures_TextOnlyVsWithExtras(t *testing.T) {
	p := newProcessorSHA256(t, sha256ConfigUnified("0", 16))
	tokens := makeTokensSHA256(16) // one block

	// Text-only (nil extras)
	keysText, err := p.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", nil)
	require.NoError(t, err)
	require.Len(t, keysText, 1)

	// With LoRA extra
	extras := []*kvblock.BlockExtraFeatures{
		{LoraName: "my-lora"},
	}
	keysLora, err := p.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", extras)
	require.NoError(t, err)
	require.Len(t, keysLora, 1)

	assert.NotEqual(t, keysText[0], keysLora[0],
		"text-only and LoRA-tagged blocks must produce different hashes")
}

func TestSHA256_ExtraFeatures_DifferentLoRAsDifferentHashes(t *testing.T) {
	p := newProcessorSHA256(t, sha256ConfigUnified("0", 16))
	tokens := makeTokensSHA256(16)

	extrasA := []*kvblock.BlockExtraFeatures{{LoraName: "lora-A"}}
	extrasB := []*kvblock.BlockExtraFeatures{{LoraName: "lora-B"}}

	keysA, err := p.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", extrasA)
	require.NoError(t, err)
	keysB, err := p.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", extrasB)
	require.NoError(t, err)

	assert.NotEqual(t, keysA[0], keysB[0],
		"different LoRA adapters must produce different hashes")
}

func TestSHA256_ExtraFeatures_DifferentMMHashesDifferentKeys(t *testing.T) {
	p := newProcessorSHA256(t, sha256ConfigUnified("0", 16))
	tokens := makeTokensSHA256(16)

	extrasA := []*kvblock.BlockExtraFeatures{
		{MMHashes: []kvblock.MMHash{{Hash: "image_A"}}},
	}
	extrasB := []*kvblock.BlockExtraFeatures{
		{MMHashes: []kvblock.MMHash{{Hash: "image_B"}}},
	}

	keysA, err := p.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", extrasA)
	require.NoError(t, err)
	keysB, err := p.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", extrasB)
	require.NoError(t, err)

	assert.NotEqual(t, keysA[0], keysB[0],
		"different MM hashes must produce different block keys")
}

func TestSHA256_ExtraFeatures_CacheSaltAffectsHash(t *testing.T) {
	p := newProcessorSHA256(t, sha256ConfigUnified("0", 16))
	tokens := makeTokensSHA256(16)

	extrasNoSalt := []*kvblock.BlockExtraFeatures{{}}
	extrasWithSalt := []*kvblock.BlockExtraFeatures{{CacheSalt: "session-123"}}

	keysNoSalt, err := p.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", extrasNoSalt)
	require.NoError(t, err)
	keysWithSalt, err := p.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", extrasWithSalt)
	require.NoError(t, err)

	assert.NotEqual(t, keysNoSalt[0], keysWithSalt[0],
		"cache salt must affect block hash")
}

func TestSHA256_ExtraFeatures_PromptEmbedsAffectHash(t *testing.T) {
	p := newProcessorSHA256(t, sha256ConfigUnified("0", 16))
	tokens := makeTokensSHA256(16)

	embedsA := []byte{0x01, 0x02, 0x03}
	embedsB := []byte{0x04, 0x05, 0x06}

	extrasA := []*kvblock.BlockExtraFeatures{{PromptEmbedsHash: embedsA}}
	extrasB := []*kvblock.BlockExtraFeatures{{PromptEmbedsHash: embedsB}}

	keysA, err := p.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", extrasA)
	require.NoError(t, err)
	keysB, err := p.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", extrasB)
	require.NoError(t, err)

	assert.NotEqual(t, keysA[0], keysB[0],
		"different prompt embeds hashes must produce different block keys")
}

func TestSHA256_ExtraFeatures_MultiBlockWithVaryingExtras(t *testing.T) {
	p := newProcessorSHA256(t, sha256ConfigUnified("0", 16))
	tokens := makeTokensSHA256(48) // 3 blocks

	// Block 0: LoRA + MM + salt
	// Block 1: LoRA + MM (no salt)
	// Block 2: LoRA only
	extras := []*kvblock.BlockExtraFeatures{
		{
			LoraName:  "adapter-1",
			MMHashes:  []kvblock.MMHash{{Hash: "img_abc"}},
			CacheSalt: "salt-first-block",
		},
		{
			LoraName: "adapter-1",
			MMHashes: []kvblock.MMHash{{Hash: "img_abc"}},
		},
		{
			LoraName: "adapter-1",
		},
	}

	keys, err := p.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", extras)
	require.NoError(t, err)
	require.Len(t, keys, 3)

	// All three blocks should have different hashes due to different extras
	assert.NotEqual(t, keys[0], keys[1], "block 0 and 1 should differ (salt)")
	assert.NotEqual(t, keys[1], keys[2], "block 1 and 2 should differ (MM hash)")
	assert.NotEqual(t, keys[0], keys[2], "block 0 and 2 should differ")
}

func TestSHA256_ExtraFeatures_RawExtrasPassthrough(t *testing.T) {
	p := newProcessorSHA256(t, sha256ConfigUnified("0", 16))
	tokens := makeTokensSHA256(16)

	// Use RawExtras to pass through exact wire format
	rawExtras := []any{"lora-name", "mm_hash_1", "salt-value"}
	extras := []*kvblock.BlockExtraFeatures{
		{RawExtras: rawExtras},
	}

	keys, err := p.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", extras)
	require.NoError(t, err)
	require.Len(t, keys, 1)

	// Verify it produces a valid hash (non-zero)
	assert.NotEqual(t, kvblock.EmptyBlockHash, keys[0])
}

func TestSHA256_ExtraFeatures_MismatchedLengthReturnsError(t *testing.T) {
	p := newProcessorSHA256(t, sha256ConfigUnified("0", 16))
	tokens := makeTokensSHA256(32) // 2 blocks

	// Only 1 extra feature for 2 blocks
	extras := []*kvblock.BlockExtraFeatures{{LoraName: "lora"}}

	_, err := p.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", extras)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "does not match token chunk count")
}

// ---------------------------------------------------------------------------
// Unified Config Tests (HashSeed + BlockSizeTokens for SHA256)
// ---------------------------------------------------------------------------

func TestSHA256_UnifiedConfig_HashSeedWorks(t *testing.T) {
	tokens := makeTokensSHA256(16)

	p1 := newProcessorSHA256(t, sha256ConfigUnified("seed-A", 16))
	p2 := newProcessorSHA256(t, sha256ConfigUnified("seed-B", 16))

	keys1, err := p1.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", nil)
	require.NoError(t, err)
	keys2, err := p2.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", nil)
	require.NoError(t, err)

	assert.NotEqual(t, keys1[0], keys2[0],
		"different HashSeed values must produce different hashes")
}

func TestSHA256_UnifiedConfig_BlockSizeTokensWorks(t *testing.T) {
	tokens := makeTokensSHA256(64)

	p16 := newProcessorSHA256(t, sha256ConfigUnified("0", 16))
	p32 := newProcessorSHA256(t, sha256ConfigUnified("0", 32))

	keys16, err := p16.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", nil)
	require.NoError(t, err)
	keys32, err := p32.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", nil)
	require.NoError(t, err)

	assert.Len(t, keys16, 4, "16-token blocks should produce 4 blocks")
	assert.Len(t, keys32, 2, "32-token blocks should produce 2 blocks")
	assert.NotEqual(t, keys16[0], keys32[0],
		"different block sizes must produce different first block hashes")
}

func TestSHA256_UnifiedConfig_DefaultsToFNV(t *testing.T) {
	// Default config should use FNV, not SHA256
	p := newProcessorSHA256(t, nil)
	tokens := makeTokensSHA256(16)

	keysFNV, err := p.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", nil)
	require.NoError(t, err)

	// Explicitly SHA256
	pSHA := newProcessorSHA256(t, sha256ConfigUnified("", 16))
	keysSHA, err := pSHA.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", nil)
	require.NoError(t, err)

	assert.NotEqual(t, keysFNV[0], keysSHA[0],
		"default (FNV) and explicit SHA256 must produce different hashes")
}

// ---------------------------------------------------------------------------
// ParseRawExtraKeys with New Fields
// ---------------------------------------------------------------------------

func TestParseRawExtraKeys_PreservesRawExtras(t *testing.T) {
	raw := [][]any{
		{"lora-name", "mm_hash_1", "salt"},
		{"lora-name", "mm_hash_2"},
	}

	result, err := kvblock.ParseRawExtraKeys(raw)
	require.NoError(t, err)
	require.Len(t, result, 2)

	// RawExtras should be preserved
	require.NotNil(t, result[0])
	assert.Equal(t, raw[0], result[0].RawExtras)

	require.NotNil(t, result[1])
	assert.Equal(t, raw[1], result[1].RawExtras)
}

func TestParseRawExtraKeys_AlsoPopulatesMMHashes(t *testing.T) {
	raw := [][]any{
		{"string_hash_1", "string_hash_2"},
	}

	result, err := kvblock.ParseRawExtraKeys(raw)
	require.NoError(t, err)
	require.NotNil(t, result[0])

	// RawExtras preserved
	assert.Equal(t, raw[0], result[0].RawExtras)

	// MMHashes also populated for legacy callers
	require.Len(t, result[0].MMHashes, 2)
	assert.Equal(t, "string_hash_1", result[0].MMHashes[0].Hash)
	assert.Equal(t, "string_hash_2", result[0].MMHashes[1].Hash)
}

func TestParseRawExtraKeys_EmptySliceReturnsNil(t *testing.T) {
	raw := [][]any{
		{}, // empty slice
	}

	result, err := kvblock.ParseRawExtraKeys(raw)
	require.NoError(t, err)
	require.Len(t, result, 1)
	assert.Nil(t, result[0], "empty inner slice should produce nil entry")
}

// ---------------------------------------------------------------------------
// Reference Value Tests with Extras
// ---------------------------------------------------------------------------

func referenceHashSHA256WithExtras(parent []byte, tokens []uint32, extras any) []byte {
	enc, err := cbor.CanonicalEncOptions().EncMode()
	if err != nil {
		panic(err)
	}
	payload := []any{parent, tokens, extras}
	b, err := enc.Marshal(payload)
	if err != nil {
		panic(err)
	}
	sum := sha256.Sum256(b)
	return sum[:]
}

func TestSHA256_ExtraFeatures_MatchesReferenceWithLoRA(t *testing.T) {
	const seed = "0"
	const blockSize = 16
	tokens := makeTokensSHA256(blockSize)

	p := newProcessorSHA256(t, sha256ConfigUnified(seed, blockSize))
	extras := []*kvblock.BlockExtraFeatures{
		{LoraName: "test-lora"},
	}

	keys, err := p.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", extras)
	require.NoError(t, err)
	require.Len(t, keys, 1)

	// Compute reference
	initDigest := referenceInitHashSHA256(seed)
	extrasArray := []any{"test-lora"}
	blockDigest := referenceHashSHA256WithExtras(initDigest, tokens, extrasArray)
	wantKey := kvblock.BlockHash(referenceEngineKey(blockDigest))

	assert.Equal(t, wantKey, keys[0],
		"SHA256 with LoRA extra must match independent reference computation")
}

func TestSHA256_ExtraFeatures_MatchesReferenceWithMultipleExtras(t *testing.T) {
	const seed = "0"
	const blockSize = 16
	tokens := makeTokensSHA256(blockSize)

	p := newProcessorSHA256(t, sha256ConfigUnified(seed, blockSize))
	extras := []*kvblock.BlockExtraFeatures{
		{
			LoraName:  "lora-1",
			MMHashes:  []kvblock.MMHash{{Hash: "mm_a"}, {Hash: "mm_b"}},
			CacheSalt: "salt-123",
		},
	}

	keys, err := p.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", extras)
	require.NoError(t, err)
	require.Len(t, keys, 1)

	// Compute reference with same ordering
	initDigest := referenceInitHashSHA256(seed)
	extrasArray := []any{"lora-1", "mm_a", "mm_b", "salt-123"}
	blockDigest := referenceHashSHA256WithExtras(initDigest, tokens, extrasArray)
	wantKey := kvblock.BlockHash(referenceEngineKey(blockDigest))

	assert.Equal(t, wantKey, keys[0],
		"SHA256 with multiple extras must match independent reference computation")
}

// Made with Bob
