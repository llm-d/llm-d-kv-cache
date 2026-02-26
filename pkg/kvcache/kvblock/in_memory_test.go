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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	. "github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/utils/logging"
)

// createInMemoryIndexForTesting creates a new InMemoryIndex for testing.
func createInMemoryIndexForTesting(t *testing.T) Index {
	t.Helper()
	cfg := DefaultInMemoryIndexConfig()
	// Set PodCacheSize to 500 to accommodate testConcurrentOperations
	// (100 goroutines * 4 pods each = 400 max concurrent pods)
	cfg.PodCacheSize = 500
	index, err := NewInMemoryIndex(cfg)
	require.NoError(t, err)
	return index
}

func TestInMemoryIndexBehavior(t *testing.T) {
	testCommonIndexBehavior(t, createInMemoryIndexForTesting)
}

func TestInMemoryIndexSize(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())

	// Test with small size to verify eviction
	cfg := &InMemoryIndexConfig{
		Size:         2, // Only 2 keys max
		PodCacheSize: 1, // Pod cache size doesn't matter for this test
	}

	index, err := NewInMemoryIndex(cfg)
	require.NoError(t, err)

	// Add first key
	engineKey1 := BlockHash(72735753)
	requestKey1 := BlockHash(79215516)
	err = index.Add(ctx, []BlockHash{engineKey1}, []BlockHash{requestKey1}, []PodEntry{{PodIdentifier: "pod1", DeviceTier: "gpu"}})
	require.NoError(t, err)

	// Add second key
	engineKey2 := BlockHash(41341092)
	requestKey2 := BlockHash(12871930)
	err = index.Add(ctx, []BlockHash{engineKey2}, []BlockHash{requestKey2}, []PodEntry{{PodIdentifier: "pod2", DeviceTier: "gpu"}})
	require.NoError(t, err)

	// Add third key - should evict the first one due to LRU
	engineKey3 := BlockHash(34012886)
	requestKey3 := BlockHash(69914638)
	err = index.Add(ctx, []BlockHash{engineKey3}, []BlockHash{requestKey3}, []PodEntry{{PodIdentifier: "pod3", DeviceTier: "cpu"}})
	require.NoError(t, err)

	// Lookup should only return the last two keys
	podsPerKey, err := index.Lookup(ctx, []BlockHash{requestKey1, requestKey2, requestKey3}, nil)
	require.NoError(t, err)

	assert.Len(t, podsPerKey, 2) // Only key2 and key3 should be present
	assert.Len(t, podsPerKey[requestKey2], 1)
	assert.Len(t, podsPerKey[requestKey3], 1)
	assert.Contains(t, podsPerKey[requestKey2], PodEntry{PodIdentifier: "pod2", DeviceTier: "gpu"})
	assert.Contains(t, podsPerKey[requestKey3], PodEntry{PodIdentifier: "pod3", DeviceTier: "cpu"})
}

func TestInMemoryIndexPodCacheSize(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())

	// Test with small limits to verify enforcement
	cfg := &InMemoryIndexConfig{
		Size:         1, // Only 1 key max
		PodCacheSize: 2, // Only 2 pods per key
	}

	index, err := NewInMemoryIndex(cfg)
	require.NoError(t, err)

	// Test PodCacheSize limit: add more pods than the limit for one key
	engineKey := BlockHash(28409753)
	requestKey := BlockHash(51374550)
	pods := []PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu"},
		{PodIdentifier: "pod2", DeviceTier: "gpu"},
		{PodIdentifier: "pod3", DeviceTier: "cpu"}, // This should evict pod1 due to LRU
	}

	err = index.Add(ctx, []BlockHash{engineKey}, []BlockHash{requestKey}, pods)
	require.NoError(t, err)

	// Lookup should only return 2 pods (pod2 and pod3), pod1 should be evicted
	podsPerKey, err := index.Lookup(ctx, []BlockHash{requestKey}, nil)
	require.NoError(t, err)
	assert.Len(t, podsPerKey, 1)
	assert.Len(t, podsPerKey[requestKey], 2, "Should only have 2 pods due to PodCacheSize limit")
	assert.Contains(t, podsPerKey[requestKey], PodEntry{PodIdentifier: "pod2", DeviceTier: "gpu"})
	assert.Contains(t, podsPerKey[requestKey], PodEntry{PodIdentifier: "pod3", DeviceTier: "cpu"})
}
