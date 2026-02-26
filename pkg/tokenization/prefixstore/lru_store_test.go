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

package prefixstore //nolint:testpackage // convenience

import (
	"strings"
	"testing"

	preprocessing "github.com/llm-d/llm-d-kv-cache/pkg/preprocessing/chat_completions"
	"github.com/stretchr/testify/assert"
)

//nolint:gocritic
func setupTestLRUTokenStore(t *testing.T, blockSize int) (*LRUTokenStore, string, []uint32) {
	t.Helper()
	store, err := NewLRUTokenStore(&Config{
		&LRUStoreConfig{CacheSize: defaultMaxCacheSize, BlockSize: blockSize},
	})
	assert.NoError(t, err)

	text := "The capital of France is Paris"
	tokens := []uint32{1, 2, 3, 4, 5, 6}
	offsets := []preprocessing.Offset{
		{0, 3}, {4, 11}, {12, 14}, {15, 21}, {22, 24}, {25, 30},
	}

	err = store.AddTokenization(text, tokens, offsets)
	assert.NoError(t, err)

	lru, ok := store.(*LRUTokenStore)
	assert.True(t, ok)
	return lru, text, tokens
}

func TestLRUTokenStore_AddAndRetrieve(t *testing.T) {
	lruStore, _, _ := setupTestLRUTokenStore(t, 4)

	prompt := "The capital of F"
	actualTokens, overlapRatio := lruStore.FindLongestContainedTokens(prompt)
	assert.Equal(t, []uint32{1, 2, 3}, actualTokens, "FindLongestContainedTokens(%q) result mismatch", prompt)
	assert.Equal(t, 1.0, overlapRatio, "FindLongestContainedTokens(%q) overlapRatio should be 1.0", prompt)
}

func TestLRUTokenStore_PartialMismatch(t *testing.T) {
	blockSize := 4
	lruStore, _, expectedTokens := setupTestLRUTokenStore(t, blockSize)

	cases := []struct {
		prompt        string
		matchPrompt   string
		tokenContains int
	}{
		{
			prompt:        "The capital of France is Marseille",
			matchPrompt:   "The capital of France is",
			tokenContains: 4,
		},
		{
			prompt:        "The capital of Japan is Tokyo",
			matchPrompt:   "The capital ",
			tokenContains: 2,
		},
		{
			prompt:        "The capital of F",
			matchPrompt:   "The capital of F",
			tokenContains: 3,
		},
		{
			prompt:        "The capital of France is Paris",
			matchPrompt:   "The capital of France is Par",
			tokenContains: 5,
		},
	}
	for _, c := range cases {
		numerator := len(c.matchPrompt)
		denominator := len(c.prompt)
		expectedRatio := float64(numerator) / float64(denominator)
		actualTokens, overlapRatio := lruStore.FindLongestContainedTokens(c.prompt)
		t.Logf("prompt: %q, actualTokens: %v, overlapRatio: %v\n", c.prompt, actualTokens, overlapRatio)
		assert.Subset(t, expectedTokens, actualTokens, "prompt=%q, result should be subset of tokens", c.prompt)
		assert.LessOrEqual(t, c.tokenContains, len(actualTokens), "prompt=%q, token count mismatch", c.prompt)
		assert.Equal(t, expectedRatio, overlapRatio, "prompt=%q, overlapRatio mismatch", c.prompt)
	}
}

func TestLRUTokenStore_PrefixMatch(t *testing.T) {
	blockSize := 4
	lruStore, text, expectedTokens := setupTestLRUTokenStore(t, blockSize)

	words := strings.Split(text, " ")
	prefix := ""
	for i, word := range words {
		if i > 0 {
			prefix += " "
		}
		prefix += word

		actualTokens, overlapRatio := lruStore.FindLongestContainedTokens(prefix)
		t.Logf("word: %q, prefix: %q, actualTokens: %v, overlapRatio: %v", word, prefix, actualTokens, overlapRatio)
		assert.Subset(t, expectedTokens, actualTokens, "prefix=%q, result should be subset of tokens", prefix)

		b := len(prefix) / blockSize
		minVal := float64(b*blockSize) / float64(len(prefix))
		assert.GreaterOrEqual(t, len(actualTokens), i, "prefix=%q, token count should be >= i", prefix)
		assert.LessOrEqual(t, len(actualTokens), i+1, "prefix=%q, token count should be <= i+1", prefix)
		assert.GreaterOrEqual(t, overlapRatio, minVal, "prefix=%q, overlapRatio should be >= min", prefix)
	}
}

func TestLRUTokenStore_LRUEviction(t *testing.T) {
	cfg := &Config{
		&LRUStoreConfig{CacheSize: 2, BlockSize: 18}, // Small cache size for testing eviction
	}
	store, err := NewLRUTokenStore(cfg)
	assert.NoError(t, err)

	texts := []string{
		"abcdefghjiklmno",
		"123456789011121314",
		"pqrstuvwxyz,./';lp",
	}
	tokens := [][]uint32{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}
	offsets := [][]preprocessing.Offset{
		{{0, 5}, {6, 10}, {11, 15}},
		{{0, 6}, {7, 12}, {13, 18}},
		{{0, 6}, {7, 12}, {13, 18}},
	}

	// Add tokenizations to the store
	for i, text := range texts {
		err = store.AddTokenization(text, tokens[i], offsets[i])
		assert.NoError(t, err)
	}

	// First text block should be evicted
	prompt := "abcdefghjiklmno"
	result, _ := store.FindLongestContainedTokens(prompt)
	assert.Empty(t, result, "First text block should be evicted")

	// Third text block should still be in cache
	prompt = "pqrstuvwxyz,./';lp"
	result, _ = store.FindLongestContainedTokens(prompt)
	assert.Equal(t, []uint32{7, 8, 9}, result)
}
