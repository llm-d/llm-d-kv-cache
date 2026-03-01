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
}

// testBasicAddAndLookup tests basic Add and Lookup functionality.
func testBasicAddAndLookup(t *testing.T, ctx context.Context, index Index) {
	t.Helper()
	engineKey := BlockHash(55269488)
	requestKey := BlockHash(10633516)
	entries := []PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu", DataParallelRank: NoDataParallelRank},
		{PodIdentifier: "pod2", DeviceTier: "gpu", DataParallelRank: NoDataParallelRank},
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
		{PodIdentifier: "pod1", DeviceTier: "gpu", DataParallelRank: NoDataParallelRank},
		{PodIdentifier: "pod2", DeviceTier: "gpu", DataParallelRank: NoDataParallelRank},
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
		{PodIdentifier: "pod1", DeviceTier: "gpu", DataParallelRank: NoDataParallelRank},
		{PodIdentifier: "pod2", DeviceTier: "gpu", DataParallelRank: NoDataParallelRank},
	}

	err := index.Add(ctx, []BlockHash{engineKey}, []BlockHash{requestKey}, entries1)
	require.NoError(t, err)

	// Second batch with one duplicate pod but different tier
	entries2 := []PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu", DataParallelRank: NoDataParallelRank}, // Same pod, same tier
		{PodIdentifier: "pod2", DeviceTier: "cpu", DataParallelRank: NoDataParallelRank}, // Same pod, different tier
		{PodIdentifier: "pod3", DeviceTier: "gpu", DataParallelRank: NoDataParallelRank},
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
		{PodIdentifier: "pod1", DeviceTier: "gpu", DataParallelRank: NoDataParallelRank},
		{PodIdentifier: "pod2", DeviceTier: "gpu", DataParallelRank: NoDataParallelRank},
		{PodIdentifier: "pod2", DeviceTier: "cpu", DataParallelRank: NoDataParallelRank},
		{PodIdentifier: "pod3", DeviceTier: "gpu", DataParallelRank: NoDataParallelRank},
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
		{PodIdentifier: "pod1", DeviceTier: "gpu", DataParallelRank: NoDataParallelRank},
		{PodIdentifier: "pod2", DeviceTier: "gpu", DataParallelRank: NoDataParallelRank},
		{PodIdentifier: "pod3", DeviceTier: "gpu", DataParallelRank: NoDataParallelRank},
	}

	err := index.Add(ctx, []BlockHash{engineKey}, []BlockHash{requestKey}, entries)
	require.NoError(t, err)

	// Lookup with filter - should only return pod1
	filterSet := sets.New("pod1")
	podsPerKey, err := index.Lookup(ctx, []BlockHash{requestKey}, filterSet)
	require.NoError(t, err)
	assert.Len(t, podsPerKey, 1)
	assert.Contains(t, podsPerKey, requestKey)
	assert.Equal(t, []PodEntry{{PodIdentifier: "pod1", DeviceTier: "gpu", DataParallelRank: NoDataParallelRank}}, podsPerKey[requestKey])

	// Lookup with multiple filters
	filterSet = sets.New("pod1", "pod3")
	podsPerKey, err = index.Lookup(ctx, []BlockHash{requestKey}, filterSet)
	require.NoError(t, err)
	assert.Len(t, podsPerKey, 1)
	assert.ElementsMatch(t, podsPerKey[requestKey], []PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu", DataParallelRank: NoDataParallelRank},
		{PodIdentifier: "pod3", DeviceTier: "gpu", DataParallelRank: NoDataParallelRank},
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
		{PodIdentifier: "pod1", DeviceTier: "gpu", DataParallelRank: NoDataParallelRank},
		{PodIdentifier: "pod2", DeviceTier: "gpu", DataParallelRank: NoDataParallelRank},
		{PodIdentifier: "pod3", DeviceTier: "gpu", DataParallelRank: NoDataParallelRank},
	}

	// Add entries
	err := index.Add(ctx, []BlockHash{engineKey}, []BlockHash{requestKey}, entries)
	require.NoError(t, err)

	// Evict specific pod entries (note: eviction is based on pod identifier only)
	evictEntries := []PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu", DataParallelRank: NoDataParallelRank},
		{PodIdentifier: "pod3", DeviceTier: "cpu", DataParallelRank: NoDataParallelRank}, // Device tier may differ from stored entry
	}

	err = index.Evict(ctx, engineKey, evictEntries)
	require.NoError(t, err)

	// Verify that pod1 was evicted but pod2 and pod3 remain
	// Note: pod3 remains because eviction only matched pod identifier, not device tier
	podsPerKey, err := index.Lookup(ctx, []BlockHash{requestKey}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Len(t, podsPerKey, 1)
	assert.Contains(t, podsPerKey, requestKey)
	expected := []PodEntry{
		{PodIdentifier: "pod2", DeviceTier: "gpu", DataParallelRank: NoDataParallelRank},
		{PodIdentifier: "pod3", DeviceTier: "gpu", DataParallelRank: NoDataParallelRank},
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
					entries := []PodEntry{{PodIdentifier: fmt.Sprintf("pod-%d-%d", id, operationIndex), DeviceTier: "gpu", DataParallelRank: NoDataParallelRank}}
					if err := index.Add(ctx, []BlockHash{engineKey}, []BlockHash{requestKey}, entries); err != nil {
						errChan <- err
					}
				case 1: // Lookup
					_, err := index.Lookup(ctx, []BlockHash{requestKey}, sets.Set[string]{})
					if err != nil {
						errChan <- err
					}
				case 2: // Evict
					entries := []PodEntry{{PodIdentifier: fmt.Sprintf("pod-%d-%d", id, operationIndex-2), DeviceTier: "gpu", DataParallelRank: NoDataParallelRank}}
					if err := index.Evict(ctx, engineKey, entries); err != nil {
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

// TestPodEntryString tests the String() method with and without DP rank.
func TestPodEntryString(t *testing.T) {
	tests := []struct {
		name     string
		entry    PodEntry
		expected string
	}{
		{
			name:     "no DP rank",
			entry:    PodEntry{PodIdentifier: "pod-1", DeviceTier: "gpu", DataParallelRank: NoDataParallelRank},
			expected: "pod-1@gpu",
		},
		{
			name:     "DP rank 0",
			entry:    PodEntry{PodIdentifier: "pod-1", DeviceTier: "gpu", DataParallelRank: 0},
			expected: "pod-1@gpu@dp0",
		},
		{
			name:     "DP rank 3",
			entry:    PodEntry{PodIdentifier: "pod-1", DeviceTier: "cpu", DataParallelRank: 3},
			expected: "pod-1@cpu@dp3",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.expected, tc.entry.String())
		})
	}
}

// TestParsePodEntry tests parsing PodEntry from string representation.
func TestParsePodEntry(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected PodEntry
		wantErr  bool
	}{
		{
			name:     "2-part format (no DP)",
			input:    "pod-1@gpu",
			expected: PodEntry{PodIdentifier: "pod-1", DeviceTier: "gpu", DataParallelRank: NoDataParallelRank},
		},
		{
			name:     "3-part format (with DP rank 0)",
			input:    "pod-1@gpu@dp0",
			expected: PodEntry{PodIdentifier: "pod-1", DeviceTier: "gpu", DataParallelRank: 0},
		},
		{
			name:     "3-part format (with DP rank 5)",
			input:    "pod-1@cpu@dp5",
			expected: PodEntry{PodIdentifier: "pod-1", DeviceTier: "cpu", DataParallelRank: 5},
		},
		{
			name:    "invalid format",
			input:   "nope",
			wantErr: true,
		},
		{
			name:     "non-dp suffix treated as 2-part",
			input:    "pod@gpu@xx0",
			expected: PodEntry{PodIdentifier: "pod@gpu", DeviceTier: "xx0", DataParallelRank: NoDataParallelRank},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			entry, err := ParsePodEntry(tc.input)
			if tc.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.Equal(t, tc.expected, entry)
		})
	}
}

// TestPodEntryRoundTrip tests that String() → ParsePodEntry() round-trips correctly.
func TestPodEntryRoundTrip(t *testing.T) {
	entries := []PodEntry{
		{PodIdentifier: "pod-1", DeviceTier: "gpu", DataParallelRank: NoDataParallelRank},
		{PodIdentifier: "pod-2", DeviceTier: "cpu", DataParallelRank: 0},
		{PodIdentifier: "pod-3", DeviceTier: "gpu", DataParallelRank: 7},
	}

	for _, original := range entries {
		t.Run(original.String(), func(t *testing.T) {
			parsed, err := ParsePodEntry(original.String())
			require.NoError(t, err)
			assert.Equal(t, original, parsed)
		})
	}
}

// TestNewPodEntry tests the NewPodEntry helper function.
func TestNewPodEntry(t *testing.T) {
	// nil rank
	entry := NewPodEntry("pod-1", "gpu", nil)
	assert.Equal(t, NoDataParallelRank, entry.DataParallelRank)

	// non-nil rank
	rank := 2
	entry = NewPodEntry("pod-1", "gpu", &rank)
	assert.Equal(t, 2, entry.DataParallelRank)
}

// TestDPAwareIndexOperations tests that the index correctly distinguishes
// between different DP ranks on the same pod.
func TestDPAwareIndexOperations(t *testing.T) {
	t.Helper()
	logger := logging.NewTestLogger().V(logging.DEBUG)
	ctx := log.IntoContext(t.Context(), logger)

	cfg := DefaultInMemoryIndexConfig()
	index, err := NewInMemoryIndex(cfg)
	require.NoError(t, err)

	engineKey1 := BlockHash(100)
	requestKey1 := BlockHash(200)
	engineKey2 := BlockHash(101)
	requestKey2 := BlockHash(201)

	// Add block from pod-1 DP rank 0
	err = index.Add(ctx,
		[]BlockHash{engineKey1}, []BlockHash{requestKey1},
		[]PodEntry{{PodIdentifier: "pod-1", DeviceTier: "gpu", DataParallelRank: 0}})
	require.NoError(t, err)

	// Add same request key from pod-1 DP rank 1
	err = index.Add(ctx,
		[]BlockHash{engineKey1}, []BlockHash{requestKey1},
		[]PodEntry{{PodIdentifier: "pod-1", DeviceTier: "gpu", DataParallelRank: 1}})
	require.NoError(t, err)

	// Add different block from pod-1 DP rank 0 only
	err = index.Add(ctx,
		[]BlockHash{engineKey2}, []BlockHash{requestKey2},
		[]PodEntry{{PodIdentifier: "pod-1", DeviceTier: "gpu", DataParallelRank: 0}})
	require.NoError(t, err)

	// Lookup requestKey1 - should find both ranks
	podsPerKey, err := index.Lookup(ctx, []BlockHash{requestKey1}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Len(t, podsPerKey[requestKey1], 2)
	assert.Contains(t, podsPerKey[requestKey1], PodEntry{PodIdentifier: "pod-1", DeviceTier: "gpu", DataParallelRank: 0})
	assert.Contains(t, podsPerKey[requestKey1], PodEntry{PodIdentifier: "pod-1", DeviceTier: "gpu", DataParallelRank: 1})

	// Lookup requestKey2 - should find only rank 0
	podsPerKey, err = index.Lookup(ctx, []BlockHash{requestKey2}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Len(t, podsPerKey[requestKey2], 1)
	assert.Contains(t, podsPerKey[requestKey2], PodEntry{PodIdentifier: "pod-1", DeviceTier: "gpu", DataParallelRank: 0})

	// Evict rank 1 from requestKey1
	err = index.Evict(ctx, engineKey1, []PodEntry{{PodIdentifier: "pod-1", DeviceTier: "gpu", DataParallelRank: 1}})
	require.NoError(t, err)

	// Verify only rank 0 remains
	podsPerKey, err = index.Lookup(ctx, []BlockHash{requestKey1}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Len(t, podsPerKey[requestKey1], 1)
	assert.Contains(t, podsPerKey[requestKey1], PodEntry{PodIdentifier: "pod-1", DeviceTier: "gpu", DataParallelRank: 0})
}
