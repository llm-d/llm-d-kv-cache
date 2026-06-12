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

// Tests for the SHA256-CBOR hashing path in TokensToKVBlockKeys.

package kvblock_test

import (
	"crypto/sha256"
	"encoding/binary"
	"sync"
	"testing"

	"github.com/fxamacker/cbor/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
)

// sha256Config returns a TokenProcessorConfig that explicitly selects SHA256-CBOR
// with the given seed and block size (for tests that need non-default values).
func sha256Config(seed string, blockSize int) *kvblock.TokenProcessorConfig {
	return &kvblock.TokenProcessorConfig{
		HashAlgorithm:   kvblock.HashAlgorithmSHA256CBOR,
		HashSeed:        seed,
		BlockSizeTokens: blockSize,
	}
}

// defaultSHA256Config returns a processor config using the default SHA256 settings
// (seed "10", block size 64), matching vLLM.
func defaultSHA256Config() *kvblock.TokenProcessorConfig {
	return sha256Config("10", 64)
}

// fnvConfig returns a TokenProcessorConfig that explicitly selects FNV-64a.
func fnvConfig(seed string, blockSize int) *kvblock.TokenProcessorConfig {
	return &kvblock.TokenProcessorConfig{
		HashAlgorithm:   kvblock.HashAlgorithmFNV64a,
		HashSeed:        seed,
		BlockSizeTokens: blockSize,
	}
}

// referenceInitHashSHA256 computes the expected seed digest the same way
// getInitHashSHA256 does, for independent verification in tests.
func referenceInitHashSHA256(seed string) []byte {
	enc, err := cbor.CanonicalEncOptions().EncMode()
	if err != nil {
		panic(err)
	}
	b, err := enc.Marshal(seed)
	if err != nil {
		panic(err)
	}
	sum := sha256.Sum256(b)
	return sum[:]
}

// referenceHashSHA256 mirrors hashSHA256 for use in reference computations.
func referenceHashSHA256(parent []byte, tokens []uint32) []byte {
	enc, err := cbor.CanonicalEncOptions().EncMode()
	if err != nil {
		panic(err)
	}
	payload := []interface{}{parent, tokens, []interface{}(nil)}
	b, err := enc.Marshal(payload)
	if err != nil {
		panic(err)
	}
	sum := sha256.Sum256(b)
	return sum[:]
}

func referenceEngineKey(digest []byte) uint64 {
	return binary.BigEndian.Uint64(digest[24:])
}

func makeTokensSHA256(n int) []uint32 {
	t := make([]uint32, n)
	for i := range t {
		t[i] = uint32(i + 1)
	}
	return t
}

func newProcessorSHA256(t *testing.T, cfg *kvblock.TokenProcessorConfig) kvblock.TokenProcessor {
	t.Helper()
	p, err := kvblock.NewChunkedTokenDatabase(cfg)
	require.NoError(t, err)
	return p
}

// --- Default-algorithm test ---

// TestSHA256_ExplicitAlgorithm verifies that explicitly setting SHA256-CBOR
// produces consistent results. Note: default algorithm is FNV64a, not SHA256.
func TestSHA256_ExplicitAlgorithm(t *testing.T) {
	p1 := newProcessorSHA256(t, &kvblock.TokenProcessorConfig{
		HashAlgorithm:   kvblock.HashAlgorithmSHA256CBOR,
		HashSeed:        "10",
		BlockSizeTokens: 64,
	})
	p2 := newProcessorSHA256(t, &kvblock.TokenProcessorConfig{
		HashAlgorithm:   kvblock.HashAlgorithmSHA256CBOR,
		HashSeed:        "10",
		BlockSizeTokens: 64,
	})

	tokens := makeTokensSHA256(64)
	keys1, err := p1.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", nil)
	require.NoError(t, err)
	keys2, err := p2.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", nil)
	require.NoError(t, err)

	assert.Equal(t, keys1, keys2, "explicit SHA256 configs with same settings must produce same result")
}

// --- Block count tests ---

// TestSHA256_BlockCount verifies chunking uses SHA256BlockSize, not BlockSizeTokens.
func TestSHA256_BlockCount(t *testing.T) {
	cases := []struct {
		name       string
		numTokens  int
		blockSize  int
		wantBlocks int
	}{
		{"zero tokens", 0, 64, 0},
		{"fewer than one block", 63, 64, 0},
		{"exactly one block", 64, 64, 1},
		{"one full + partial", 100, 64, 1},
		{"exactly two blocks", 128, 64, 2},
		{"three blocks", 192, 64, 3},
		{"large input", 640, 64, 10},
		{"custom block size 32", 128, 32, 4},
		{"custom block size 16", 64, 16, 4},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := newProcessorSHA256(t, sha256Config("10", tc.blockSize))
			tokens := makeTokensSHA256(tc.numTokens)
			keys, err := p.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", nil)
			require.NoError(t, err)
			assert.Len(t, keys, tc.wantBlocks)
		})
	}
}

// --- Determinism tests ---

func TestSHA256_Deterministic(t *testing.T) {
	p := newProcessorSHA256(t, defaultSHA256Config())
	tokens := makeTokensSHA256(192) // 3 blocks

	var first []kvblock.BlockHash
	for i := 0; i < 5; i++ {
		keys, err := p.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", nil)
		require.NoError(t, err)
		require.Len(t, keys, 3)
		if i == 0 {
			first = keys
		} else {
			assert.Equal(t, first, keys, "keys must be identical on repeated calls (iter %d)", i)
		}
	}
}

// TestSHA256_SeedDrivesOutput verifies that different SHA256HashSeed values produce
// different keys (seed is respected by the SHA256 path).
func TestSHA256_SeedDrivesOutput(t *testing.T) {
	tokens := makeTokensSHA256(64)

	seeds := []string{"10", "0", "abc", "different"}
	seen := map[kvblock.BlockHash]string{}
	for _, seed := range seeds {
		p := newProcessorSHA256(t, sha256Config(seed, 64))
		keys, err := p.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", nil)
		require.NoError(t, err)
		require.Len(t, keys, 1)
		if prev, exists := seen[keys[0]]; exists {
			t.Errorf("seed collision: seeds %q and %q produced the same key", seed, prev)
		}
		seen[keys[0]] = seed
	}
}

// TestSHA256_BlockSizeDrivesOutput verifies that different SHA256BlockSize values
// produce different keys (chunking boundary changes the hash chain).
func TestSHA256_BlockSizeDrivesOutput(t *testing.T) {
	tokens := makeTokensSHA256(128)

	p32 := newProcessorSHA256(t, sha256Config("10", 32))
	p64 := newProcessorSHA256(t, sha256Config("10", 64))

	keys32, err := p32.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", nil)
	require.NoError(t, err)
	keys64, err := p64.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", nil)
	require.NoError(t, err)

	require.Len(t, keys32, 4)
	require.Len(t, keys64, 2)
	assert.NotEqual(t, keys32[0], keys64[0], "different block sizes must produce different first keys")
}

// --- Reference value tests ---

// TestSHA256_FirstBlockMatchesReference verifies the first key against an
// independently computed SHA256-CBOR reference, confirming vLLM compatibility.
func TestSHA256_FirstBlockMatchesReference(t *testing.T) {
	const seed = "10"
	const blockSize = 64
	tokens := makeTokensSHA256(blockSize)

	p := newProcessorSHA256(t, sha256Config(seed, blockSize))
	keys, err := p.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", nil)
	require.NoError(t, err)
	require.Len(t, keys, 1)

	initDigest := referenceInitHashSHA256(seed)
	blockDigest := referenceHashSHA256(initDigest, tokens)
	wantKey := kvblock.BlockHash(referenceEngineKey(blockDigest))

	assert.Equal(t, wantKey, keys[0], "first block key must match independent SHA256-CBOR reference")
}

// TestSHA256_ChainedBlocksMatchReference verifies all blocks in a multi-block
// sequence match the expected chained SHA256-CBOR computation.
func TestSHA256_ChainedBlocksMatchReference(t *testing.T) {
	const seed = "10"
	const blockSize = 64
	const numBlocks = 4
	tokens := makeTokensSHA256(blockSize * numBlocks)

	p := newProcessorSHA256(t, sha256Config(seed, blockSize))
	keys, err := p.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", nil)
	require.NoError(t, err)
	require.Len(t, keys, numBlocks)

	digest := referenceInitHashSHA256(seed)
	for i := 0; i < numBlocks; i++ {
		chunk := tokens[i*blockSize : (i+1)*blockSize]
		digest = referenceHashSHA256(digest, chunk)
		wantKey := kvblock.BlockHash(referenceEngineKey(digest))
		assert.Equal(t, wantKey, keys[i], "block %d key mismatch", i)
	}
}

// --- Guard and edge-case tests ---

// TestSHA256_NonEmptyParentReturnsNil verifies that a non-empty parentKey is
// rejected (returns nil, no error).
func TestSHA256_NonEmptyParentReturnsNil(t *testing.T) {
	p := newProcessorSHA256(t, defaultSHA256Config())
	tokens := makeTokensSHA256(64)

	keys, err := p.TokensToKVBlockKeys(kvblock.BlockHash(999), tokens, "model", nil)
	require.NoError(t, err)
	assert.Nil(t, keys, "non-empty parentKey must return nil for sha256_cbor path")
}

func TestSHA256_EmptyAndPartialTokens(t *testing.T) {
	p := newProcessorSHA256(t, defaultSHA256Config())

	keys, err := p.TokensToKVBlockKeys(kvblock.EmptyBlockHash, nil, "model", nil)
	require.NoError(t, err)
	assert.Empty(t, keys, "nil tokens must produce no keys")

	keys, err = p.TokensToKVBlockKeys(kvblock.EmptyBlockHash, makeTokensSHA256(63), "model", nil)
	require.NoError(t, err)
	assert.Empty(t, keys, "partial block (< blockSize tokens) must produce no keys")
}

// --- Uniqueness and independence tests ---

func TestSHA256_DifferentTokensDifferentKeys(t *testing.T) {
	p := newProcessorSHA256(t, defaultSHA256Config())

	a := makeTokensSHA256(64)
	b := makeTokensSHA256(64)
	b[0] = 99999

	keysA, err := p.TokensToKVBlockKeys(kvblock.EmptyBlockHash, a, "model", nil)
	require.NoError(t, err)
	keysB, err := p.TokensToKVBlockKeys(kvblock.EmptyBlockHash, b, "model", nil)
	require.NoError(t, err)

	require.Len(t, keysA, 1)
	require.Len(t, keysB, 1)
	assert.NotEqual(t, keysA[0], keysB[0])
}

func TestSHA256_KeysAreNotEmptyBlockHash(t *testing.T) {
	p := newProcessorSHA256(t, defaultSHA256Config())
	keys, err := p.TokensToKVBlockKeys(kvblock.EmptyBlockHash, makeTokensSHA256(320), "model", nil)
	require.NoError(t, err)
	for i, k := range keys {
		assert.NotEqual(t, kvblock.EmptyBlockHash, k, "block %d must not be EmptyBlockHash", i)
	}
}

// TestSHA256_ModelNameIgnored verifies that modelName has no effect (sha256_cbor
// does not include modelName in the hash chain).
func TestSHA256_ModelNameIgnored(t *testing.T) {
	p := newProcessorSHA256(t, defaultSHA256Config())
	tokens := makeTokensSHA256(64)

	var first []kvblock.BlockHash
	for _, m := range []string{"gpt-4", "llama-3", "", "any-model"} {
		keys, err := p.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, m, nil)
		require.NoError(t, err)
		if first == nil {
			first = keys
		} else {
			assert.Equal(t, first, keys, "modelName %q must not affect sha256_cbor keys", m)
		}
	}
}

// TestSHA256_ExtraFeaturesAffectHash verifies that extraFeatures DO affect the hash
// in SHA256-CBOR mode (this is the key difference from FNV64a mode).
func TestSHA256_ExtraFeaturesAffectHash(t *testing.T) {
	p := newProcessorSHA256(t, defaultSHA256Config())
	tokens := makeTokensSHA256(64)

	keysNil, err := p.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", nil)
	require.NoError(t, err)

	extra := []*kvblock.BlockExtraFeatures{{MMHashes: []kvblock.MMHash{{Hash: "12345"}, {Hash: "67890"}}}}
	keysExtra, err := p.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", extra)
	require.NoError(t, err)

	assert.NotEqual(t, keysNil, keysExtra, "extraFeatures MUST affect sha256_cbor output (vLLM compatibility)")
}

// --- Concurrency test ---

func TestSHA256_ConcurrentCalls(t *testing.T) {
	p := newProcessorSHA256(t, defaultSHA256Config())
	tokens := makeTokensSHA256(192)

	reference, err := p.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", nil)
	require.NoError(t, err)

	const goroutines = 50
	results := make(chan []kvblock.BlockHash, goroutines)
	var wg sync.WaitGroup
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			keys, err := p.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", nil)
			if err == nil {
				results <- keys
			}
		}()
	}
	wg.Wait()
	close(results)

	count := 0
	for keys := range results {
		count++
		assert.Equal(t, reference, keys, "concurrent call produced different keys")
	}
	assert.Equal(t, goroutines, count)
}

// --- BlockSize() method ---

func TestSHA256_BlockSizeMethodReflectsConfig(t *testing.T) {
	p := newProcessorSHA256(t, &kvblock.TokenProcessorConfig{
		HashAlgorithm:   kvblock.HashAlgorithmSHA256CBOR,
		HashSeed:        "10",
		BlockSizeTokens: 32, // BlockSize() returns this
	})
	assert.Equal(t, 32, p.BlockSize(), "BlockSize() must return BlockSizeTokens")
}

// --- Algorithm isolation test ---

// TestAlgorithmIsolation verifies that SHA256 and FNV produce different keys
// for the same token input (they must not be accidentally interchangeable).
func TestAlgorithmIsolation(t *testing.T) {
	tokens := makeTokensSHA256(64)

	pSHA := newProcessorSHA256(t, sha256Config("10", 64))
	pFNV := newProcessorSHA256(t, fnvConfig("", 64))

	keysSHA, err := pSHA.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", nil)
	require.NoError(t, err)
	keysFNV, err := pFNV.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", nil)
	require.NoError(t, err)

	require.Len(t, keysSHA, 1)
	require.Len(t, keysFNV, 1)
	assert.NotEqual(t, keysSHA[0], keysFNV[0], "SHA256 and FNV must produce different keys for the same input")
}
