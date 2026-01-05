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
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/controller-runtime/pkg/log"

	. "github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/utils/logging"
)

// testCommonIndexBehavior runs a comprehensive test suite for any Index implementation.
// indexFactory should return a fresh index instance for each test to ensure test isolation.
func testCommonIndexBehavior(t *testing.T, indexFactory func(t *testing.T) Index) {
	t.Helper()
	logger := logging.NewTestLogger().V(logging.DEBUG)
	ctx := log.IntoContext(t.Context(), logger)

	t.Run("BasicAddAndLookup", func(t *testing.T) {
		index := indexFactory(t)
		testBasicAddAndLookup(t, ctx, index)
	})

	t.Run("DuplicatePodHandling", func(t *testing.T) {
		index := indexFactory(t)
		testDuplicatePodHandling(t, ctx, index)
	})

	t.Run("FilteredLookup", func(t *testing.T) {
		index := indexFactory(t)
		testFilteredLookup(t, ctx, index)
	})

	t.Run("EvictBasic", func(t *testing.T) {
		index := indexFactory(t)
		testEvictBasic(t, ctx, index)
	})

	t.Run("ConcurrentOperations", func(t *testing.T) {
		index := indexFactory(t)
		testConcurrentOperations(t, ctx, index)
	})

	t.Run("SameHashDifferentModels", func(t *testing.T) {
		index := indexFactory(t)
		testSameHashDifferentModels(t, ctx, index)
	})
}

// testBasicAddAndLookup tests basic Add and Lookup functionality.
func testBasicAddAndLookup(t *testing.T, ctx context.Context, index Index) {
	t.Helper()
	engineKey := Key{ModelName: "test-model", ChunkHash: 55269488}
	requestKey := Key{ModelName: "test-model", ChunkHash: 10633516}
	entries := []PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu"},
		{PodIdentifier: "pod2", DeviceTier: "gpu"},
	}

	// Add entries
	err := index.Add(ctx, []Key{engineKey}, []Key{requestKey}, entries)
	require.NoError(t, err)

	// Lookup all entries
	podsPerKey, err := index.Lookup(ctx, []Key{requestKey}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Len(t, podsPerKey, 1)
	assert.Contains(t, podsPerKey, requestKey)
	assert.ElementsMatch(t, podsPerKey[requestKey], []PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu"},
		{PodIdentifier: "pod2", DeviceTier: "gpu"},
	})
}

// testDuplicatePodHandling tests behavior when adding duplicate pod identifiers.
// The current implementation allows duplicate pod identifiers with different device tiers,
// treating them as separate entries in the index.
func testDuplicatePodHandling(t *testing.T, ctx context.Context, index Index) {
	t.Helper()
	engineKey := Key{ModelName: "test-model", ChunkHash: 91642125}
	requestKey := Key{ModelName: "test-model", ChunkHash: 61519471}

	// First batch of entries
	entries1 := []PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu"},
		{PodIdentifier: "pod2", DeviceTier: "gpu"},
	}

	err := index.Add(ctx, []Key{engineKey}, []Key{requestKey}, entries1)
	require.NoError(t, err)

	// Second batch with one duplicate pod but different tier
	entries2 := []PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu"}, // Same pod, same tier
		{PodIdentifier: "pod2", DeviceTier: "cpu"}, // Same pod, different tier
		{PodIdentifier: "pod3", DeviceTier: "gpu"},
	}

	err = index.Add(ctx, []Key{engineKey}, []Key{requestKey}, entries2)
	require.NoError(t, err)

	// Lookup and verify the behavior with duplicates
	// Note: The index currently preserves duplicate pod identifiers as separate entries
	podsPerKey, err := index.Lookup(ctx, []Key{requestKey}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Len(t, podsPerKey, 1)
	assert.Contains(t, podsPerKey, requestKey)

	// Should contain all pod entries, including duplicates with different tiers
	// Expected: pod1(gpu), pod2(gpu), pod2(cpu), pod3(gpu)
	expected := []PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu"},
		{PodIdentifier: "pod2", DeviceTier: "gpu"},
		{PodIdentifier: "pod2", DeviceTier: "cpu"},
		{PodIdentifier: "pod3", DeviceTier: "gpu"},
	}
	assert.ElementsMatch(t, podsPerKey[requestKey], expected)
}

// testFilteredLookup tests lookup with pod identifier filtering.
// This verifies that the index can filter results based on specific pod identifiers.
func testFilteredLookup(t *testing.T, ctx context.Context, index Index) {
	t.Helper()
	engineKey := Key{ModelName: "test-model", ChunkHash: 93788608}
	requestKey := Key{ModelName: "test-model", ChunkHash: 55204205}
	entries := []PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu"},
		{PodIdentifier: "pod2", DeviceTier: "gpu"},
		{PodIdentifier: "pod3", DeviceTier: "gpu"},
	}

	err := index.Add(ctx, []Key{engineKey}, []Key{requestKey}, entries)
	require.NoError(t, err)

	// Lookup with filter - should only return pod1
	filterSet := sets.New("pod1")
	podsPerKey, err := index.Lookup(ctx, []Key{requestKey}, filterSet)
	require.NoError(t, err)
	assert.Len(t, podsPerKey, 1)
	assert.Contains(t, podsPerKey, requestKey)
	assert.Equal(t, []PodEntry{{PodIdentifier: "pod1", DeviceTier: "gpu"}}, podsPerKey[requestKey])

	// Lookup with multiple filters
	filterSet = sets.New("pod1", "pod3")
	podsPerKey, err = index.Lookup(ctx, []Key{requestKey}, filterSet)
	require.NoError(t, err)
	assert.Len(t, podsPerKey, 1)
	assert.ElementsMatch(t, podsPerKey[requestKey], []PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu"},
		{PodIdentifier: "pod3", DeviceTier: "gpu"},
	})

	// Lookup with non-existent pod filter should return empty result
	filterSet = sets.New("pod999")
	podsPerKey, err = index.Lookup(ctx, []Key{requestKey}, filterSet)
	require.NoError(t, err)
	assert.Len(t, podsPerKey, 0) // No matching pods found
}

// testEvictBasic tests basic eviction functionality.
// Verifies that specific pod entries can be removed from the index.
func testEvictBasic(t *testing.T, ctx context.Context, index Index) {
	t.Helper()
	engineKey := Key{ModelName: "test-model", ChunkHash: 17434655}
	requestKey := Key{ModelName: "test-model", ChunkHash: 59244875}
	entries := []PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu"},
		{PodIdentifier: "pod2", DeviceTier: "gpu"},
		{PodIdentifier: "pod3", DeviceTier: "gpu"},
	}

	// Add entries
	err := index.Add(ctx, []Key{engineKey}, []Key{requestKey}, entries)
	require.NoError(t, err)

	// Evict specific pod entries (note: eviction is based on pod identifier only)
	evictEntries := []PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu"},
		{PodIdentifier: "pod3", DeviceTier: "cpu"}, // Device tier may differ from stored entry
	}

	err = index.Evict(ctx, engineKey, evictEntries)
	require.NoError(t, err)

	// Verify that pod1 was evicted but pod2 and pod3 remain
	// Note: pod3 remains because eviction only matched pod identifier, not device tier
	podsPerKey, err := index.Lookup(ctx, []Key{requestKey}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Len(t, podsPerKey, 1)
	assert.Contains(t, podsPerKey, requestKey)
	expected := []PodEntry{
		{PodIdentifier: "pod2", DeviceTier: "gpu"},
		{PodIdentifier: "pod3", DeviceTier: "gpu"},
	}
	assert.ElementsMatch(t, expected, podsPerKey[requestKey])
}

// testConcurrentOperations tests thread safety with concurrent operations.
func testConcurrentOperations(t *testing.T, ctx context.Context, index Index) {
	t.Helper()
	engineKey := Key{ModelName: "test-model", ChunkHash: 38894120}
	requestKey := Key{ModelName: "test-model", ChunkHash: 72568158}

	var wg sync.WaitGroup
	errChan := make(chan error, 1000)

	// Run 100 goroutines doing concurrent operations
	for goroutineID := 0; goroutineID < 100; goroutineID++ {
		wg.Add(1)
		go func(id int) {
			time.Sleep(time.Millisecond * time.Duration(id%10)) // Stagger start times
			defer wg.Done()
			for operationIndex := 0; operationIndex < 10; operationIndex++ {
				switch operationIndex % 3 {
				case 0: // Add
					entries := []PodEntry{{PodIdentifier: fmt.Sprintf("pod-%d-%d", id, operationIndex), DeviceTier: "gpu"}}
					if err := index.Add(ctx, []Key{engineKey}, []Key{requestKey}, entries); err != nil {
						errChan <- err
					}
				case 1: // Lookup
					podsPerKey, err := index.Lookup(ctx, []Key{requestKey}, sets.Set[string]{})
					if err != nil {
						errChan <- err
					}
					assert.Contains(t, podsPerKey, requestKey)
					expectedPod := PodEntry{
						PodIdentifier: fmt.Sprintf("pod-%d-%d", id, operationIndex-1),
						DeviceTier:    "gpu",
					}
					assert.Contains(t, podsPerKey[requestKey], expectedPod)
				case 2: // Evict
					entries := []PodEntry{{PodIdentifier: fmt.Sprintf("pod-%d-%d", id, operationIndex-2), DeviceTier: "gpu"}}
					if err := index.Evict(ctx, engineKey, entries); err != nil {
						errChan <- err
					}
					podsPerKey, err := index.Lookup(ctx, []Key{requestKey}, sets.Set[string]{})
					if err != nil {
						errChan <- err
					}
					if _, ok := podsPerKey[requestKey]; ok {
						evictedPod := PodEntry{
							PodIdentifier: fmt.Sprintf("pod-%d-%d", id, operationIndex-2),
							DeviceTier:    "gpu",
						}
						assert.NotContains(t, podsPerKey[requestKey], evictedPod)
					}
				}
			}
		}(goroutineID)
	}

	wg.Wait()
	close(errChan)

	// Check for errors
	for err := range errChan {
		require.NoError(t, err)
	}

	// Verify index still works
	_, err := index.Lookup(ctx, []Key{requestKey}, sets.Set[string]{})
	require.NoError(t, err)
}

// testSameHashDifferentModels tests that keys with the same hash but different model names
// are treated as distinct entries to avoid collisions between models.
func testSameHashDifferentModels(t *testing.T, ctx context.Context, index Index) {
	t.Helper()
	// Use the same hash value for different models
	sameHash := uint64(12345678)

	// Create keys with same hash but different models
	engineKey1 := Key{ModelName: "baseModel", ChunkHash: sameHash}
	requestKey1 := Key{ModelName: "baseModel", ChunkHash: sameHash}

	engineKey2 := Key{ModelName: "loraAdapter", ChunkHash: sameHash}
	requestKey2 := Key{ModelName: "loraAdapter", ChunkHash: sameHash}

	// Add entries for first model
	entries1 := []PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu"},
		{PodIdentifier: "pod2", DeviceTier: "gpu"},
	}
	err := index.Add(ctx, []Key{engineKey1}, []Key{requestKey1}, entries1)
	require.NoError(t, err)

	// Add entries for second model
	entries2 := []PodEntry{
		{PodIdentifier: "pod3", DeviceTier: "gpu"},
		{PodIdentifier: "pod4", DeviceTier: "cpu"},
	}
	err = index.Add(ctx, []Key{engineKey2}, []Key{requestKey2}, entries2)
	require.NoError(t, err)

	// Lookup entries for first model - should only return pods for that model
	podsPerKey, err := index.Lookup(ctx, []Key{requestKey1}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Len(t, podsPerKey, 1)
	assert.Contains(t, podsPerKey, requestKey1)
	assert.ElementsMatch(t, podsPerKey[requestKey1], entries1)

	// Lookup entries for second model - should only return pods for that model
	podsPerKey, err = index.Lookup(ctx, []Key{requestKey2}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Len(t, podsPerKey, 1)
	assert.Contains(t, podsPerKey, requestKey2)
	assert.ElementsMatch(t, podsPerKey[requestKey2], entries2)

	// Lookup both keys together - should return separate entries for each model
	podsPerKey, err = index.Lookup(ctx, []Key{requestKey1, requestKey2}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Len(t, podsPerKey, 2)
	assert.Contains(t, podsPerKey, requestKey1)
	assert.Contains(t, podsPerKey, requestKey2)
	assert.ElementsMatch(t, podsPerKey[requestKey1], entries1)
	assert.ElementsMatch(t, podsPerKey[requestKey2], entries2)

	// Evict from first model - should not affect second model
	evictEntries := []PodEntry{{PodIdentifier: "pod1", DeviceTier: "gpu"}}
	err = index.Evict(ctx, engineKey1, evictEntries)
	require.NoError(t, err)

	// Verify first model has one less entry
	podsPerKey, err = index.Lookup(ctx, []Key{requestKey1}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Len(t, podsPerKey, 1)
	expected1 := []PodEntry{{PodIdentifier: "pod2", DeviceTier: "gpu"}}
	assert.ElementsMatch(t, podsPerKey[requestKey1], expected1)

	// Verify second model is unaffected
	podsPerKey, err = index.Lookup(ctx, []Key{requestKey2}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Len(t, podsPerKey, 1)
	assert.ElementsMatch(t, podsPerKey[requestKey2], entries2)
}
