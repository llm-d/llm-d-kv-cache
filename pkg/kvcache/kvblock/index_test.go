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

	t.Run("AddWithNilEngineKeys", func(t *testing.T) {
		index := indexFactory(t)
		testAddWithNilEngineKeys(t, ctx, index)
	})

	t.Run("StressConcurrentAddOverlappingKeys", func(t *testing.T) {
		index := indexFactory(t)
		testStressConcurrentAddOverlappingKeys(t, ctx, index)
	})

	t.Run("StressConcurrentAddEvictInterleaved", func(t *testing.T) {
		index := indexFactory(t)
		testStressConcurrentAddEvictInterleaved(t, ctx, index)
	})

	t.Run("StressConcurrentAddLookup", func(t *testing.T) {
		index := indexFactory(t)
		testStressConcurrentAddLookup(t, ctx, index)
	})

	t.Run("StressConcurrentEvictDuringLookup", func(t *testing.T) {
		index := indexFactory(t)
		testStressConcurrentEvictDuringLookup(t, ctx, index)
	})

	t.Run("StressHighCardinality", func(t *testing.T) {
		index := indexFactory(t)
		testStressHighCardinality(t, ctx, index)
	})
}

// testBasicAddAndLookup tests basic Add and Lookup functionality.
func testBasicAddAndLookup(t *testing.T, ctx context.Context, index Index) {
	t.Helper()
	engineKey := BlockHash(55269488)
	requestKey := BlockHash(10633516)
	entries := []PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu"},
		{PodIdentifier: "pod2", DeviceTier: "gpu"},
	}

	// Add entries
	err := index.Add(ctx, []BlockHash{engineKey}, []BlockHash{requestKey}, entries)
	require.NoError(t, err)

	// Lookup all entries
	podsPerKey, err := index.Lookup(ctx, []BlockHash{requestKey}, sets.Set[string]{})
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
	engineKey := BlockHash(91642125)
	requestKey := BlockHash(61519471)

	// First batch of entries
	entries1 := []PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu"},
		{PodIdentifier: "pod2", DeviceTier: "gpu"},
	}

	err := index.Add(ctx, []BlockHash{engineKey}, []BlockHash{requestKey}, entries1)
	require.NoError(t, err)

	// Second batch with one duplicate pod but different tier
	entries2 := []PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu"}, // Same pod, same tier
		{PodIdentifier: "pod2", DeviceTier: "cpu"}, // Same pod, different tier
		{PodIdentifier: "pod3", DeviceTier: "gpu"},
	}

	err = index.Add(ctx, []BlockHash{engineKey}, []BlockHash{requestKey}, entries2)
	require.NoError(t, err)

	// Lookup and verify the behavior with duplicates
	// Note: The index currently preserves duplicate pod identifiers as separate entries
	podsPerKey, err := index.Lookup(ctx, []BlockHash{requestKey}, sets.Set[string]{})
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
	engineKey := BlockHash(93788608)
	requestKey := BlockHash(55204205)
	entries := []PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu"},
		{PodIdentifier: "pod2", DeviceTier: "gpu"},
		{PodIdentifier: "pod3", DeviceTier: "gpu"},
	}

	err := index.Add(ctx, []BlockHash{engineKey}, []BlockHash{requestKey}, entries)
	require.NoError(t, err)

	// Lookup with filter - should only return pod1
	filterSet := sets.New("pod1")
	podsPerKey, err := index.Lookup(ctx, []BlockHash{requestKey}, filterSet)
	require.NoError(t, err)
	assert.Len(t, podsPerKey, 1)
	assert.Contains(t, podsPerKey, requestKey)
	assert.Equal(t, []PodEntry{{PodIdentifier: "pod1", DeviceTier: "gpu"}}, podsPerKey[requestKey])

	// Lookup with multiple filters
	filterSet = sets.New("pod1", "pod3")
	podsPerKey, err = index.Lookup(ctx, []BlockHash{requestKey}, filterSet)
	require.NoError(t, err)
	assert.Len(t, podsPerKey, 1)
	assert.ElementsMatch(t, podsPerKey[requestKey], []PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu"},
		{PodIdentifier: "pod3", DeviceTier: "gpu"},
	})

	// Lookup with non-existent pod filter should return empty result
	filterSet = sets.New("pod999")
	podsPerKey, err = index.Lookup(ctx, []BlockHash{requestKey}, filterSet)
	require.NoError(t, err)
	assert.Len(t, podsPerKey, 0) // No matching pods found
}

// testEvictBasic tests basic eviction functionality.
// Verifies that specific pod entries can be removed from the index.
func testEvictBasic(t *testing.T, ctx context.Context, index Index) {
	t.Helper()
	engineKey := BlockHash(17434655)
	requestKey := BlockHash(59244875)
	entries := []PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu"},
		{PodIdentifier: "pod2", DeviceTier: "gpu"},
		{PodIdentifier: "pod3", DeviceTier: "gpu"},
	}

	// Add entries
	err := index.Add(ctx, []BlockHash{engineKey}, []BlockHash{requestKey}, entries)
	require.NoError(t, err)

	// Evict specific pod entries (note: eviction is based on pod identifier only)
	evictEntries := []PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu"},
		{PodIdentifier: "pod3", DeviceTier: "cpu"}, // Device tier may differ from stored entry
	}

	err = index.Evict(ctx, engineKey, EngineKey, evictEntries)
	require.NoError(t, err)

	// Verify that pod1 was evicted but pod2 and pod3 remain
	// Note: pod3 remains because eviction only matched pod identifier, not device tier
	podsPerKey, err := index.Lookup(ctx, []BlockHash{requestKey}, sets.Set[string]{})
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
	engineKey := BlockHash(38894120)
	requestKey := BlockHash(72568158)

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
					if err := index.Add(ctx, []BlockHash{engineKey}, []BlockHash{requestKey}, entries); err != nil {
						errChan <- err
					}
				case 1: // Lookup
					_, err := index.Lookup(ctx, []BlockHash{requestKey}, sets.Set[string]{})
					if err != nil {
						errChan <- err
					}
				case 2: // Evict
					entries := []PodEntry{{PodIdentifier: fmt.Sprintf("pod-%d-%d", id, operationIndex-2), DeviceTier: "gpu"}}
					if err := index.Evict(ctx, engineKey, EngineKey, entries); err != nil {
						errChan <- err
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
	_, err := index.Lookup(ctx, []BlockHash{requestKey}, sets.Set[string]{})
	require.NoError(t, err)
}

// testStressConcurrentAddOverlappingKeys runs N goroutines all adding to the
// same set of keys, verifying no panics, errors, or corrupted state.
func testStressConcurrentAddOverlappingKeys(t *testing.T, ctx context.Context, index Index) {
	t.Helper()
	const numGoroutines = 100
	const numKeys = 10

	requestKeys := make([]BlockHash, numKeys)
	engineKeys := make([]BlockHash, numKeys)
	for i := range numKeys {
		requestKeys[i] = BlockHash(uint64(8000000) + uint64(i)) // #nosec G115 -- test data, i is small
		engineKeys[i] = BlockHash(uint64(9000000) + uint64(i))  // #nosec G115 -- test data, i is small
	}

	var wg sync.WaitGroup
	errChan := make(chan error, numGoroutines*numKeys)

	for g := range numGoroutines {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for k := range numKeys {
				entry := PodEntry{PodIdentifier: fmt.Sprintf("pod-%d", id), DeviceTier: "gpu"}
				if err := index.Add(ctx, []BlockHash{engineKeys[k]}, []BlockHash{requestKeys[k]}, []PodEntry{entry}); err != nil {
					errChan <- fmt.Errorf("goroutine %d key %d: Add failed: %w", id, k, err)
				}
			}
		}(g)
	}

	wg.Wait()
	close(errChan)
	for err := range errChan {
		require.NoError(t, err)
	}

	podsPerKey, err := index.Lookup(ctx, requestKeys, sets.Set[string]{})
	require.NoError(t, err)
	for _, rk := range requestKeys {
		assert.NotEmpty(t, podsPerKey[rk], "key %v should have entries after concurrent adds", rk)
	}
}

// testStressConcurrentAddEvictInterleaved runs Add and Evict for the same keys
// simultaneously, verifying no deadlocks or panics.
func testStressConcurrentAddEvictInterleaved(t *testing.T, ctx context.Context, index Index) {
	t.Helper()
	const numGoroutines = 50
	const numIterations = 20

	engineKey := BlockHash(7100000)
	requestKey := BlockHash(7200000)

	// Seed the index so evictions have something to remove.
	seed := PodEntry{PodIdentifier: "seed-pod", DeviceTier: "gpu"}
	require.NoError(t, index.Add(ctx, []BlockHash{engineKey}, []BlockHash{requestKey}, []PodEntry{seed}))

	var wg sync.WaitGroup
	errChan := make(chan error, numGoroutines*numIterations*2)

	for goroutine := range numGoroutines {
		wg.Add(2)

		// Adder
		go func(id int) {
			defer wg.Done()
			for i := range numIterations {
				entry := PodEntry{PodIdentifier: fmt.Sprintf("add-%d-%d", id, i), DeviceTier: "gpu"}
				if err := index.Add(ctx, []BlockHash{engineKey}, []BlockHash{requestKey}, []PodEntry{entry}); err != nil {
					errChan <- err
				}
			}
		}(goroutine)

		// Evicter
		go func(id int) {
			defer wg.Done()
			for i := range numIterations {
				entry := PodEntry{PodIdentifier: fmt.Sprintf("add-%d-%d", id, i), DeviceTier: "gpu"}
				if err := index.Evict(ctx, engineKey, EngineKey, []PodEntry{entry}); err != nil {
					errChan <- err
				}
			}
		}(goroutine)
	}

	wg.Wait()
	close(errChan)
	for err := range errChan {
		require.NoError(t, err)
	}

	// Index should still be queryable (no deadlock, no corruption).
	_, err := index.Lookup(ctx, []BlockHash{requestKey}, sets.Set[string]{})
	require.NoError(t, err)
}

// testStressConcurrentAddLookup runs Add and Lookup in parallel, ensuring
// lookups never panic or return errors regardless of concurrent mutations.
func testStressConcurrentAddLookup(t *testing.T, ctx context.Context, index Index) {
	t.Helper()
	const numWriters = 50
	const numReaders = 50
	const numIterations = 20

	requestKeys := make([]BlockHash, 5)
	engineKeys := make([]BlockHash, 5)
	for i := range 5 {
		requestKeys[i] = BlockHash(uint64(6100000) + uint64(i)) // #nosec G115 -- test data, i is small
		engineKeys[i] = BlockHash(uint64(6200000) + uint64(i))  // #nosec G115 -- test data, i is small
	}

	var wg sync.WaitGroup
	errChan := make(chan error, (numWriters+numReaders)*numIterations)

	// Writers
	for g := range numWriters {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := range numIterations {
				k := i % len(requestKeys)
				entry := PodEntry{PodIdentifier: fmt.Sprintf("w-%d-%d", id, i), DeviceTier: "gpu"}
				if err := index.Add(ctx, []BlockHash{engineKeys[k]}, []BlockHash{requestKeys[k]}, []PodEntry{entry}); err != nil {
					errChan <- err
				}
			}
		}(g)
	}

	// Readers
	for range numReaders {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range numIterations {
				_, err := index.Lookup(ctx, requestKeys, sets.Set[string]{})
				if err != nil {
					errChan <- err
				}
			}
		}()
	}

	wg.Wait()
	close(errChan)
	for err := range errChan {
		require.NoError(t, err)
	}
}

// testStressConcurrentEvictDuringLookup populates the index then runs Evict
// and Lookup concurrently, verifying graceful handling.
func testStressConcurrentEvictDuringLookup(t *testing.T, ctx context.Context, index Index) {
	t.Helper()
	const numKeys = 20
	const numGoroutines = 50

	requestKeys := make([]BlockHash, numKeys)
	engineKeys := make([]BlockHash, numKeys)
	for i := range numKeys {
		requestKeys[i] = BlockHash(uint64(5100000) + uint64(i)) // #nosec G115 -- test data, i is small
		engineKeys[i] = BlockHash(uint64(5200000) + uint64(i))  // #nosec G115 -- test data, i is small
		entries := []PodEntry{
			{PodIdentifier: fmt.Sprintf("pod-a-%d", i), DeviceTier: "gpu"},
			{PodIdentifier: fmt.Sprintf("pod-b-%d", i), DeviceTier: "gpu"},
		}
		require.NoError(t, index.Add(ctx, []BlockHash{engineKeys[i]}, []BlockHash{requestKeys[i]}, entries))
	}

	var wg sync.WaitGroup
	errChan := make(chan error, numGoroutines*numKeys*2)

	// Evicters — remove one pod per key
	for range numGoroutines / 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for k := range numKeys {
				entry := PodEntry{PodIdentifier: fmt.Sprintf("pod-a-%d", k), DeviceTier: "gpu"}
				if err := index.Evict(ctx, engineKeys[k], EngineKey, []PodEntry{entry}); err != nil {
					errChan <- err
				}
			}
		}()
	}

	// Readers — lookup all keys simultaneously
	for range numGoroutines / 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range numKeys {
				_, err := index.Lookup(ctx, requestKeys, sets.Set[string]{})
				if err != nil {
					errChan <- err
				}
			}
		}()
	}

	wg.Wait()
	close(errChan)
	for err := range errChan {
		require.NoError(t, err)
	}

	// Final lookup must succeed (no deadlock, no corruption).
	_, err := index.Lookup(ctx, requestKeys, sets.Set[string]{})
	require.NoError(t, err)
}

// testStressHighCardinality exercises 1000+ unique keys with 100+ goroutines
// performing Add, Lookup, and Evict simultaneously.
func testStressHighCardinality(t *testing.T, ctx context.Context, index Index) {
	t.Helper()
	const numKeys = 1000
	const numGoroutines = 120

	requestKeys := make([]BlockHash, numKeys)
	engineKeys := make([]BlockHash, numKeys)
	for i := range numKeys {
		requestKeys[i] = BlockHash(uint64(1000000) + uint64(i)) // #nosec G115 -- test data, i is small
		engineKeys[i] = BlockHash(uint64(2000000) + uint64(i))  // #nosec G115 -- test data, i is small
	}

	var wg sync.WaitGroup
	errChan := make(chan error, numGoroutines*100)

	for goroutine := range numGoroutines {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := range numKeys {
				k := (id*7 + i) % numKeys // spread access across keys
				switch i % 3 {
				case 0:
					entry := PodEntry{PodIdentifier: fmt.Sprintf("hc-%d", id), DeviceTier: "gpu"}
					if err := index.Add(ctx, []BlockHash{engineKeys[k]}, []BlockHash{requestKeys[k]}, []PodEntry{entry}); err != nil {
						errChan <- err
						return
					}
				case 1:
					if _, err := index.Lookup(ctx, []BlockHash{requestKeys[k]}, sets.Set[string]{}); err != nil {
						errChan <- err
						return
					}
				case 2:
					entry := PodEntry{PodIdentifier: fmt.Sprintf("hc-%d", id), DeviceTier: "gpu"}
					if err := index.Evict(ctx, engineKeys[k], EngineKey, []PodEntry{entry}); err != nil {
						errChan <- err
						return
					}
				}
			}
		}(goroutine)
	}

	wg.Wait()
	close(errChan)
	for err := range errChan {
		require.NoError(t, err)
	}

	// Verify index is still consistent: lookup a sample of keys.
	sample := requestKeys[:10]
	podsPerKey, err := index.Lookup(ctx, sample, sets.Set[string]{})
	require.NoError(t, err)
	assert.NotNil(t, podsPerKey)
}

// testAddWithNilEngineKeys tests that Add() with nil engineKeys only creates
// requestKey -> PodEntry mappings without engineKey -> requestKey mappings.
// All Index implementations must handle nil engineKeys without panicking.
func testAddWithNilEngineKeys(t *testing.T, ctx context.Context, index Index) {
	t.Helper()
	requestKey := BlockHash(55555555)
	pod := PodEntry{PodIdentifier: "10.0.0.3:8080", Speculative: true}

	// Add with nil engineKeys should not panic or error
	err := index.Add(ctx, nil, []BlockHash{requestKey}, []PodEntry{pod})
	require.NoError(t, err)

	// Lookup by requestKey should find the entry
	podsPerKey, err := index.Lookup(ctx, []BlockHash{requestKey}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Len(t, podsPerKey[requestKey], 1)
	assert.Contains(t, podsPerKey[requestKey], pod)

	// GetRequestKey should NOT find a mapping (no engineKey was stored)
	_, err = index.GetRequestKey(ctx, requestKey)
	assert.Error(t, err, "GetRequestKey should fail since no engineKey mapping was created")
}
