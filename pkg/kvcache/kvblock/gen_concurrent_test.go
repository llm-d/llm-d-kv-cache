/*
Copyright 2025 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package kvblock

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/sets"
)

// TestSweepDoesNotEvictFreshEntries verifies that an entry re-added after Clear
// (stamped gen == current gen, so stamped < current is false) survives Sweep.
// This is the "re-add after invalidation" contract.
func TestSweepDoesNotEvictFreshEntries(t *testing.T) {
	pod := PodEntry{PodIdentifier: "pod", DeviceTier: "gpu"}
	key := BlockHash(0xDEADBEEF)
	ctx := context.Background()

	t.Run("InMemory", func(t *testing.T) {
		idx, err := NewInMemoryIndex(nil)
		require.NoError(t, err)

		require.NoError(t, idx.Add(ctx, nil, []BlockHash{key}, []PodEntry{pod}))
		require.NoError(t, idx.Clear(ctx, pod))
		// Re-add stamps gen = current (post-Clear) so condition stamped < current is false.
		require.NoError(t, idx.Add(ctx, nil, []BlockHash{key}, []PodEntry{pod}))

		removed := idx.Sweep(ctx)
		assert.Equal(t, 0, removed, "Sweep must not remove an entry added after Clear")

		hits, err := idx.Lookup(ctx, []BlockHash{key}, sets.Set[string]{})
		require.NoError(t, err)
		assert.Len(t, hits[key], 1, "re-added pod must be visible after Sweep")
	})

	t.Run("CostAwareMemory", func(t *testing.T) {
		idx, err := NewCostAwareMemoryIndex(nil)
		require.NoError(t, err)

		require.NoError(t, idx.Add(ctx, nil, []BlockHash{key}, []PodEntry{pod}))
		require.NoError(t, idx.Clear(ctx, pod))
		require.NoError(t, idx.Add(ctx, nil, []BlockHash{key}, []PodEntry{pod}))

		removed := idx.Sweep(ctx)
		assert.Equal(t, 0, removed, "Sweep must not remove an entry added after Clear")

		hits, err := idx.Lookup(ctx, []BlockHash{key}, sets.Set[string]{})
		require.NoError(t, err)
		assert.Len(t, hits[key], 1)
	})

	t.Run("Redis", func(t *testing.T) {
		idx, _, cleanup := newMiniRedisIndex(t)
		defer cleanup()

		require.NoError(t, idx.Add(ctx, nil, []BlockHash{key}, []PodEntry{pod}))
		require.NoError(t, idx.Clear(ctx, pod))
		require.NoError(t, idx.Add(ctx, nil, []BlockHash{key}, []PodEntry{pod}))

		removed, err := idx.Sweep(ctx)
		require.NoError(t, err)
		assert.Equal(t, 0, removed, "Sweep must not remove an entry added after Clear")

		hits, err := idx.Lookup(ctx, []BlockHash{key}, sets.Set[string]{})
		require.NoError(t, err)
		assert.Len(t, hits[key], 1)
	})
}

// TestConcurrentSweepAndAdd_FreshEntriesPreserved exercises the Bug 2 fix:
// Sweep must not delete a requestKey whose PodCache a concurrent Add has
// populated after the key appeared empty.
//
// Invariant: after Add(post-Clear) and Sweep both finish, the added entry is
// always reachable via Lookup (Add rebuilds the entry if Sweep removed the key
// before Add ran, or Sweep skips removal if Add already populated the cache).
func TestConcurrentSweepAndAdd_FreshEntriesPreserved(t *testing.T) {
	const rounds = 200
	pod := PodEntry{PodIdentifier: "pod", DeviceTier: "gpu"}
	ctx := context.Background()

	for round := 0; round < rounds; round++ {
		idx, err := NewInMemoryIndex(nil)
		require.NoError(t, err)

		key := BlockHash(uint64(round + 1))

		// Pre-populate so Sweep has stale entries to remove.
		require.NoError(t, idx.Add(ctx, nil, []BlockHash{key}, []PodEntry{pod}))
		require.NoError(t, idx.Clear(ctx, pod)) // gen 0→1; entry is now stale

		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			defer wg.Done()
			idx.Sweep(ctx) // removes the stale gen-0 entry, key becomes empty
		}()
		go func() {
			defer wg.Done()
			// Re-adds with stamped gen = 1 (current); must survive Sweep.
			_ = idx.Add(ctx, nil, []BlockHash{key}, []PodEntry{pod})
		}()
		wg.Wait()

		hits, err := idx.Lookup(ctx, []BlockHash{key}, sets.Set[string]{})
		require.NoError(t, err)
		assert.Len(t, hits[key], 1,
			"round %d: entry added after Clear must be visible post Sweep", round)
	}
}

// TestConcurrentClearAndAdd verifies no panics or corruption when Clear and Add
// race on the same index. Run with -race to detect data races.
func TestConcurrentClearAndAdd(t *testing.T) {
	idx, err := NewInMemoryIndex(nil)
	require.NoError(t, err)

	const goroutines = 20
	const iterations = 200
	pod := PodEntry{PodIdentifier: "pod", DeviceTier: "gpu"}
	ctx := context.Background()

	var wg sync.WaitGroup
	for g := range goroutines / 2 {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			key := BlockHash(uint64(id + 1))
			for range iterations {
				_ = idx.Add(ctx, nil, []BlockHash{key}, []PodEntry{pod})
			}
		}(g)
	}
	for range goroutines / 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range iterations {
				_ = idx.Clear(ctx, pod)
			}
		}()
	}
	wg.Wait()
}

// TestStartSweeper_ConcurrentClear checks that the eager sweepCh initialization
// (Bug 1 fix) prevents a data race between Clear and StartSweeper.
// Run with -race to verify.
func TestStartSweeper_ConcurrentClear(t *testing.T) {
	idx, err := NewInMemoryIndex(nil)
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	pod := PodEntry{PodIdentifier: "pod", DeviceTier: "gpu"}

	var wg sync.WaitGroup
	for range 100 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = idx.Clear(context.Background(), pod)
		}()
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		idx.StartSweeper(ctx, time.Millisecond)
	}()
	wg.Wait()
}

// TestMultipleClearMultiplePods verifies that Clear for one pod does not
// invalidate entries for another pod.
func TestMultipleClearMultiplePods(t *testing.T) {
	idx, err := NewInMemoryIndex(nil)
	require.NoError(t, err)

	ctx := context.Background()
	podA := PodEntry{PodIdentifier: "A", DeviceTier: "gpu"}
	podB := PodEntry{PodIdentifier: "B", DeviceTier: "gpu"}
	key := BlockHash(1)

	require.NoError(t, idx.Add(ctx, nil, []BlockHash{key}, []PodEntry{podA, podB}))
	require.NoError(t, idx.Clear(ctx, podA))

	hits, err := idx.Lookup(ctx, []BlockHash{key}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Len(t, hits[key], 1, "only podB should remain after clearing podA")
	assert.Equal(t, podB, hits[key][0])

	// Sweep should evict podA's stale entry only.
	removed := idx.Sweep(ctx)
	assert.Equal(t, 1, removed)

	hits, err = idx.Lookup(ctx, []BlockHash{key}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Len(t, hits[key], 1, "podB still present after Sweep")
}

// --- Benchmarks ---------------------------------------------------------------

// BenchmarkClear_InMemory confirms Clear is O(1) regardless of index size.
func BenchmarkClear_InMemory(b *testing.B) {
	idx, _ := NewInMemoryIndex(nil)
	ctx := context.Background()
	pod := PodEntry{PodIdentifier: "pod", DeviceTier: "gpu"}

	keys := make([]BlockHash, 10_000)
	for i := range keys {
		keys[i] = BlockHash(uint64(i + 1))
	}
	_ = idx.Add(ctx, nil, keys, []PodEntry{pod})

	b.ResetTimer()
	for range b.N {
		_ = idx.Clear(ctx, pod)
	}
}

// BenchmarkLookup_HotPath_InMemory measures the no-Clear fast path (one slice copy).
func BenchmarkLookup_HotPath_InMemory(b *testing.B) {
	idx, _ := NewInMemoryIndex(nil)
	ctx := context.Background()
	pod := PodEntry{PodIdentifier: "pod", DeviceTier: "gpu"}

	keys := make([]BlockHash, 100)
	for i := range keys {
		keys[i] = BlockHash(uint64(i + 1))
	}
	_ = idx.Add(ctx, nil, keys, []PodEntry{pod})

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, _ = idx.Lookup(ctx, keys, sets.Set[string]{})
		}
	})
}

// BenchmarkLookup_WithGenFilter_InMemory measures the per-entry generation
// filter path that activates once any Clear has occurred.
func BenchmarkLookup_WithGenFilter_InMemory(b *testing.B) {
	idx, _ := NewInMemoryIndex(nil)
	ctx := context.Background()
	podA := PodEntry{PodIdentifier: "A", DeviceTier: "gpu"}
	podB := PodEntry{PodIdentifier: "B", DeviceTier: "gpu"}

	keys := make([]BlockHash, 100)
	for i := range keys {
		keys[i] = BlockHash(uint64(i + 1))
	}
	_ = idx.Add(ctx, nil, keys, []PodEntry{podA, podB})
	// One Clear activates the gen-filter branch in Lookup.
	_ = idx.Clear(ctx, PodEntry{PodIdentifier: "other", DeviceTier: "gpu"})

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, _ = idx.Lookup(ctx, keys, sets.Set[string]{})
		}
	})
}

// BenchmarkSweep_1k_InMemory characterizes Sweep over a 1 000-entry stale index.
func BenchmarkSweep_1k_InMemory(b *testing.B) {
	ctx := context.Background()
	pod := PodEntry{PodIdentifier: "pod", DeviceTier: "gpu"}
	keys := make([]BlockHash, 1_000)
	for i := range keys {
		keys[i] = BlockHash(uint64(i + 1))
	}

	b.ResetTimer()
	for range b.N {
		b.StopTimer()
		idx, _ := NewInMemoryIndex(nil)
		_ = idx.Add(ctx, nil, keys, []PodEntry{pod})
		_ = idx.Clear(ctx, pod)
		b.StartTimer()
		idx.Sweep(ctx)
	}
}

// BenchmarkConcurrentClearLookup_InMemory simulates a realistic workload where
// most goroutines read (Lookup) while a minority write (Clear+Add).
func BenchmarkConcurrentClearLookup_InMemory(b *testing.B) {
	idx, _ := NewInMemoryIndex(nil)
	ctx := context.Background()
	pod := PodEntry{PodIdentifier: "pod", DeviceTier: "gpu"}

	keys := make([]BlockHash, 100)
	for i := range keys {
		keys[i] = BlockHash(uint64(i + 1))
	}
	_ = idx.Add(ctx, nil, keys, []PodEntry{pod})

	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			if i%10 == 0 {
				_ = idx.Clear(ctx, pod)
				_ = idx.Add(ctx, nil, keys, []PodEntry{pod})
			} else {
				_, _ = idx.Lookup(ctx, keys, sets.Set[string]{})
			}
			i++
		}
	})
}
