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

package kvblock_test

import (
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
)

func TestGetInitHash_ConsistentHashesForSameModel(t *testing.T) {
	config := &kvblock.TokenProcessorConfig{
		BlockSize: 16,
		HashSeed:  "test-seed",
	}

	processor := kvblock.NewChunkedTokenDatabase(config)

	modelName := "test-model"
	tokens := []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16} // Full block

	// Get keys multiple times with no parent (should use init hash)
	keys1 := processor.TokensToKVBlockKeys(nil, tokens, modelName)
	keys2 := processor.TokensToKVBlockKeys(nil, tokens, modelName)
	keys3 := processor.TokensToKVBlockKeys(nil, tokens, modelName)

	require.NotEmpty(t, keys1, "Should generate keys")
	require.NotEmpty(t, keys2, "Should generate keys")
	require.NotEmpty(t, keys3, "Should generate keys")

	// All first keys should be identical (derived from same init hash)
	assert.Equal(t, keys1[0].ChunkHash, keys2[0].ChunkHash, "First key hash should be consistent across calls")
	assert.Equal(t, keys1[0].ChunkHash, keys3[0].ChunkHash, "First key hash should be consistent across calls")
	assert.Equal(t, keys1[0].ModelName, modelName, "Model name should match")
	assert.NotZero(t, keys1[0].ChunkHash, "Hash should not be zero")
}

func TestGetInitHash_DifferentHashesForDifferentModels(t *testing.T) {
	config := &kvblock.TokenProcessorConfig{
		BlockSize: 16,
		HashSeed:  "test-seed",
	}

	processor := kvblock.NewChunkedTokenDatabase(config)

	// Test different model names
	models := []string{
		"gpt-4",
		"llama-2-7b",
		"claude-3",
		"gemini-pro",
		"",  // empty string
		"a", // single character
		"very-long-model-name-with-special-characters-123!@#",
	}

	tokens := []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16} // Full block
	hashes := make(map[string]uint64)

	// Get first key hash for each model (derived from init hash)
	for _, modelName := range models {
		keys := processor.TokensToKVBlockKeys(nil, tokens, modelName)
		require.NotEmpty(t, keys, "Should generate keys for model: %s", modelName)

		hash := keys[0].ChunkHash
		hashes[modelName] = hash
		assert.NotZero(t, hash, "Hash should not be zero for model: %s", modelName)
		assert.Equal(t, keys[0].ModelName, modelName, "Model name should match")
	}

	// Verify all hashes are different
	seenHashes := make(map[uint64]string)
	for modelName, hash := range hashes {
		if existingModel, exists := seenHashes[hash]; exists {
			t.Errorf("Hash collision detected: models '%s' and '%s' have the same initial key hash %d",
				modelName, existingModel, hash)
		}
		seenHashes[hash] = modelName
	}
}

func TestGetInitHash_DifferentSeedsProduceDifferentHashes(t *testing.T) {
	modelName := "test-model"
	tokens := []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}

	// Test with different seeds
	seeds := []string{
		"",
		"seed1",
		"seed2",
		"different-seed",
		"123456",
	}

	hashes := make(map[string]uint64)

	for _, seed := range seeds {
		config := &kvblock.TokenProcessorConfig{
			BlockSize: 16,
			HashSeed:  seed,
		}

		processor := kvblock.NewChunkedTokenDatabase(config)
		keys := processor.TokensToKVBlockKeys(nil, tokens, modelName)
		require.NotEmpty(t, keys, "Should generate keys for seed: %s", seed)

		hash := keys[0].ChunkHash
		hashes[seed] = hash
		assert.NotZero(t, hash, "Hash should not be zero for seed: %s", seed)
	}

	// Verify all hashes are different
	seenHashes := make(map[uint64]string)
	for seed, hash := range hashes {
		if existingSeed, exists := seenHashes[hash]; exists {
			t.Errorf("Hash collision detected: seeds '%s' and '%s' produce the same initial hash %d for model %s",
				seed, existingSeed, hash, modelName)
		}
		seenHashes[hash] = seed
	}
}

func TestGetInitHash_ConcurrentAccess(t *testing.T) {
	config := &kvblock.TokenProcessorConfig{
		BlockSize: 16,
		HashSeed:  "test-seed",
	}

	processor := kvblock.NewChunkedTokenDatabase(config)

	modelName := "concurrent-test-model"
	tokens := []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	numGoroutines := 100

	// Channel to collect results
	results := make(chan uint64, numGoroutines)
	var wg sync.WaitGroup

	// Start multiple goroutines calling TokensToKVBlockKeys (which calls getInitHash)
	for range numGoroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()
			keys := processor.TokensToKVBlockKeys(nil, tokens, modelName)
			if len(keys) > 0 {
				results <- keys[0].ChunkHash
			}
		}()
	}

	wg.Wait()
	close(results)

	// Collect all results
	hashes := make([]uint64, 0, numGoroutines)
	for hash := range results {
		hashes = append(hashes, hash)
	}

	require.Len(t, hashes, numGoroutines, "Should have received hash from all goroutines")

	// Verify all hashes are identical
	expectedHash := hashes[0]
	for i, hash := range hashes {
		assert.Equal(t, expectedHash, hash, "Hash mismatch at index %d", i)
	}

	assert.NotZero(t, expectedHash, "Hash should not be zero")
}

func TestGetInitHash_Deterministic(t *testing.T) {
	// Test that the same configuration always produces the same hash
	modelName := "deterministic-test"
	seed := "deterministic-seed"
	tokens := []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}

	var hashes []uint64

	// Create multiple instances with same config
	for i := 0; i < 5; i++ {
		config := &kvblock.TokenProcessorConfig{
			BlockSize: 16,
			HashSeed:  seed,
		}

		processor := kvblock.NewChunkedTokenDatabase(config)
		keys := processor.TokensToKVBlockKeys(nil, tokens, modelName)
		require.NotEmpty(t, keys, "Should generate keys for instance %d", i)

		hash := keys[0].ChunkHash
		hashes = append(hashes, hash)
	}

	// All instances should produce the same hash
	expectedHash := hashes[0]
	for i, hash := range hashes {
		assert.Equal(t, expectedHash, hash, "Hash should be deterministic across instances, mismatch at index %d", i)
	}

	assert.NotZero(t, expectedHash, "Hash should not be zero")
}
