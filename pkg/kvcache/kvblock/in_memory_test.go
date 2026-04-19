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
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/sets"

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

// TestSpeculativeAnnotation tests that speculative and confirmed PodEntries
// are treated as separate entries due to the Speculative field, and that
// evicting one does not affect the other.
func TestSpeculativeAnnotation(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	index := createInMemoryIndexForTesting(t)

	requestKey := BlockHash(22222222)

	t.Run("SpeculativeAddWithNilEngineKeys", func(t *testing.T) {
		// Add a speculative entry with nil engineKeys (no engineKey mapping)
		speculativePod := PodEntry{PodIdentifier: "10.0.0.1:8080", Speculative: true}
		err := index.Add(ctx, nil, []BlockHash{requestKey}, []PodEntry{speculativePod})
		require.NoError(t, err)

		// Lookup should return the speculative pod
		podsPerKey, err := index.Lookup(ctx, []BlockHash{requestKey}, nil)
		require.NoError(t, err)
		assert.Len(t, podsPerKey[requestKey], 1)
		assert.Contains(t, podsPerKey[requestKey], speculativePod)
	})

	t.Run("SpeculativeAndConfirmedCoexist", func(t *testing.T) {
		// Add a confirmed entry for the same pod (with engineKey mapping, same requestKey)
		confirmedEngineKey := BlockHash(33333333)
		confirmedPod := PodEntry{PodIdentifier: "10.0.0.1:8080"}
		err := index.Add(ctx, []BlockHash{confirmedEngineKey}, []BlockHash{requestKey}, []PodEntry{confirmedPod})
		require.NoError(t, err)

		// Both speculative and confirmed should coexist
		podsPerKey, err := index.Lookup(ctx, []BlockHash{requestKey}, nil)
		require.NoError(t, err)
		assert.Len(t, podsPerKey[requestKey], 2)
		assert.Contains(t, podsPerKey[requestKey],
			PodEntry{PodIdentifier: "10.0.0.1:8080", Speculative: true})
		assert.Contains(t, podsPerKey[requestKey], PodEntry{PodIdentifier: "10.0.0.1:8080"})
	})

	t.Run("SpeculativeEvictPreservesConfirmed", func(t *testing.T) {
		// Evict the speculative entry using requestKey directly (no engineKey mapping exists).
		speculativePod := PodEntry{PodIdentifier: "10.0.0.1:8080", Speculative: true}
		err := index.Evict(ctx, requestKey, RequestKey, []PodEntry{speculativePod})
		require.NoError(t, err)

		// Confirmed entry should remain
		podsPerKey, err := index.Lookup(ctx, []BlockHash{requestKey}, nil)
		require.NoError(t, err)
		assert.Len(t, podsPerKey[requestKey], 1)
		assert.Contains(t, podsPerKey[requestKey], PodEntry{PodIdentifier: "10.0.0.1:8080"})
		// Speculative should be gone
		assert.NotContains(t, podsPerKey[requestKey],
			PodEntry{PodIdentifier: "10.0.0.1:8080", Speculative: true})
	})
}

// TestSpeculativeEvictThenEmpty tests that when only a speculative entry exists
// (added with nil engineKeys) and is evicted, the key is cleaned up properly.
func TestSpeculativeEvictThenEmpty(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	index := createInMemoryIndexForTesting(t)

	requestKey := BlockHash(44444444)
	speculativePod := PodEntry{PodIdentifier: "10.0.0.2:8080", Speculative: true}

	// Add speculative entry with nil engineKeys (no engineKey mapping)
	err := index.Add(ctx, nil, []BlockHash{requestKey}, []PodEntry{speculativePod})
	require.NoError(t, err)

	// Verify it exists
	podsPerKey, err := index.Lookup(ctx, []BlockHash{requestKey}, nil)
	require.NoError(t, err)
	assert.Len(t, podsPerKey[requestKey], 1)

	// Evict speculative entry using requestKey directly
	err = index.Evict(ctx, requestKey, RequestKey, []PodEntry{speculativePod})
	require.NoError(t, err)

	// Lookup should return empty
	podsPerKey, err = index.Lookup(ctx, []BlockHash{requestKey}, nil)
	require.NoError(t, err)
	assert.Empty(t, podsPerKey[requestKey])
}

// TestAddWithNilEngineKeys tests that Add() with nil engineKeys only creates
// requestKey -> PodEntry mappings without engineKey -> requestKey mappings.
func TestAddWithNilEngineKeys(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	index := createInMemoryIndexForTesting(t)

	requestKey := BlockHash(55555555)
	pod := PodEntry{PodIdentifier: "10.0.0.3:8080", Speculative: true}

	// Add with nil engineKeys
	err := index.Add(ctx, nil, []BlockHash{requestKey}, []PodEntry{pod})
	require.NoError(t, err)

	// Lookup by requestKey should work
	podsPerKey, err := index.Lookup(ctx, []BlockHash{requestKey}, nil)
	require.NoError(t, err)
	assert.Len(t, podsPerKey[requestKey], 1)
	assert.Contains(t, podsPerKey[requestKey], pod)

	// GetRequestKey should NOT find a mapping (no engineKey was stored)
	_, err = index.GetRequestKey(ctx, requestKey)
	assert.Error(t, err, "GetRequestKey should fail since no engineKey mapping was created")
}

// TestConcurrentAddEvictToEmpty is a regression test for issue #421.
// It reproduces the race between a concurrent Add and an Evict that empties
// the PodCache on the same key. Without the fix, the Evict could remove the
// PodCache from the map after Add fetched it but before Add wrote into it,
// causing the newly added entries to be orphaned and lost.
func TestConcurrentAddEvictToEmpty(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())

	const iterations = 500

	for iteration := 0; iteration < iterations; iteration++ {
		cfg := DefaultInMemoryIndexConfig()
		cfg.PodCacheSize = 10
		index, err := NewInMemoryIndex(cfg)
		require.NoError(t, err)

		engineKey := BlockHash(11111111)
		requestKey := BlockHash(22222222)
		seedPod := PodEntry{PodIdentifier: "seed", DeviceTier: "gpu"}
		survivorPod := PodEntry{PodIdentifier: fmt.Sprintf("survivor-%d", iteration), DeviceTier: "gpu"}

		// Pre-populate with a single pod so Evict can empty the cache.
		err = index.Add(ctx, []BlockHash{engineKey}, []BlockHash{requestKey}, []PodEntry{seedPod})
		require.NoError(t, err)

		var wg sync.WaitGroup
		wg.Add(2)

		// Goroutine 1: Evict the seed pod, making the cache empty.
		// This triggers PodCache removal from the map.
		go func() {
			defer wg.Done()
			//nolint:errcheck // best-effort eviction in concurrent test
			index.Evict(ctx, engineKey, EngineKey, []PodEntry{seedPod})
		}()

		// Goroutine 2: Add a new pod to the same key concurrently.
		go func() {
			defer wg.Done()
			//nolint:errcheck // best-effort add in concurrent test
			index.Add(ctx, []BlockHash{engineKey}, []BlockHash{requestKey}, []PodEntry{survivorPod})
		}()

		wg.Wait()

		// The survivor pod must be findable. Before the fix, if Evict ran
		// between Add's Get and Add's write, the survivor would be written
		// into an orphaned PodCache and lost.
		podsPerKey, err := index.Lookup(ctx, []BlockHash{requestKey}, sets.Set[string]{})
		require.NoError(t, err)

		found := false
		for _, pod := range podsPerKey[requestKey] {
			if pod.PodIdentifier == survivorPod.PodIdentifier {
				found = true
				break
			}
		}
		assert.True(t, found, "iteration %d: survivor pod %q was lost — Add wrote into an orphaned PodCache",
			iteration, survivorPod.PodIdentifier)
	}
}

// TestConcurrentAddWithStaleEvicts verifies that a flood of Evicts on the same
// key cannot cause Add to spin indefinitely. This guards the prevLen > 0 check
// in Evict: an Evict that finds an already-empty PodCache must NOT mark it as
// removed, otherwise Add's retry loop would keep creating new PodCaches that
// get immediately invalidated.
func TestConcurrentAddWithStaleEvicts(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())

	cfg := DefaultInMemoryIndexConfig()
	cfg.PodCacheSize = 10
	index, err := NewInMemoryIndex(cfg)
	require.NoError(t, err)

	engineKey := BlockHash(99999999)
	requestKey := BlockHash(88888888)
	targetPod := PodEntry{PodIdentifier: "target", DeviceTier: "gpu"}
	stalePod := PodEntry{PodIdentifier: "stale", DeviceTier: "gpu"}

	// We need the engineKey→requestKey mapping to exist so Evict can resolve
	// the engineKey. Seed it with a throwaway Add, then evict the entry.
	err = index.Add(ctx, []BlockHash{engineKey}, []BlockHash{requestKey}, []PodEntry{stalePod})
	require.NoError(t, err)
	err = index.Evict(ctx, engineKey, EngineKey, []PodEntry{stalePod})
	require.NoError(t, err)
	// Re-establish the engineKey mapping (Evict removed it when cache emptied).
	err = index.Add(ctx, []BlockHash{engineKey}, []BlockHash{requestKey}, []PodEntry{stalePod})
	require.NoError(t, err)

	// Launch many goroutines that continuously Evict a pod from the same key.
	// If Evict incorrectly marks freshly-created empty PodCaches as removed,
	// the concurrent Add below would spin forever.
	const evictors = 20
	stop := make(chan struct{})
	var wg sync.WaitGroup

	for evictor := 0; evictor < evictors; evictor++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
					//nolint:errcheck // best-effort eviction in concurrent test
					index.Evict(ctx, engineKey, EngineKey, []PodEntry{stalePod})
				}
			}
		}()
	}

	// Add must complete promptly despite the flood of concurrent Evicts.
	done := make(chan struct{})
	go func() {
		//nolint:errcheck // best-effort add in concurrent test
		index.Add(ctx, []BlockHash{engineKey}, []BlockHash{requestKey}, []PodEntry{targetPod})
		close(done)
	}()

	// 2 seconds is more than enough — a non-spinning Add finishes in microseconds.
	timeout := time.NewTimer(2 * time.Second)
	defer timeout.Stop()

	select {
	case <-done:
		close(stop)
		wg.Wait()

		podsPerKey, lookupErr := index.Lookup(ctx, []BlockHash{requestKey}, sets.Set[string]{})
		require.NoError(t, lookupErr)
		assert.Contains(t, podsPerKey[requestKey], targetPod,
			"target pod must be present after Add completes")
	case <-timeout.C:
		close(stop)
		wg.Wait()
		t.Fatal("Add did not complete within 2s — likely spinning due to stale Evicts marking empty PodCaches as removed")
	}
}

func TestPodEntryString(t *testing.T) {
	confirmed := PodEntry{PodIdentifier: "10.0.0.1:8080", DeviceTier: "gpu"}
	assert.Equal(t, "10.0.0.1:8080@gpu", confirmed.String())

	speculative := PodEntry{PodIdentifier: "10.0.0.1:8080", DeviceTier: "gpu", Speculative: true}
	assert.Equal(t, "10.0.0.1:8080@gpu[speculative]", speculative.String())

	notSpeculative := PodEntry{PodIdentifier: "10.0.0.1:8080", DeviceTier: "gpu", Speculative: false}
	assert.Equal(t, "10.0.0.1:8080@gpu", notSpeculative.String())
}
