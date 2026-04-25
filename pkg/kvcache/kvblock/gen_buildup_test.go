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
	"math/rand/v2"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/sets"
)

// TestGenBuildup exercises repeated Clear+Add cycles with *fresh* request keys
// each round. The naive expectation under the generation-counter scheme without
// any cleanup is that stale entries pile up in the index until LRU/cost pressure
// reclaims them. This test verifies:
//
//  1. Without Sweep: Lookup correctly returns zero hits for cleared rounds — i.e.,
//     correctness is preserved despite stale entries lingering — and the live
//     entry count grows linearly with rounds.
//  2. With explicit Sweep: stale entries are physically removed; the live entry
//     count stays bounded at one round's worth.
//  3. With StartSweeper background goroutine: stale entries are reclaimed within
//     a few debounce cycles without explicit Sweep calls.
func TestGenBuildup(t *testing.T) {
	const (
		rounds       = 10
		keysPerRound = 500
		pod          = "podA"
		tier         = "gpu"
	)
	podEntry := PodEntry{PodIdentifier: pod, DeviceTier: tier}

	t.Run("InMemory_NoSweep_StaleAccumulates", func(t *testing.T) {
		idx, err := NewInMemoryIndex(nil)
		require.NoError(t, err)

		for round := 0; round < rounds; round++ {
			keys := keysForRound(round, keysPerRound)
			require.NoError(t, idx.Add(t.Context(), keys, keys, []PodEntry{podEntry}))

			// Sanity: the new round's entries are visible.
			hits, _ := idx.Lookup(t.Context(), keys, sets.Set[string]{})
			assert.Len(t, hits, keysPerRound, "round %d entries visible before clear", round)

			require.NoError(t, idx.Clear(t.Context(), podEntry))

			// Lookup after Clear must return zero — correctness invariant.
			postClear, _ := idx.Lookup(t.Context(), keys, sets.Set[string]{})
			assert.Empty(t, postClear, "round %d entries cleared by Clear", round)
		}

		// Without Sweep, the index physically still holds entries for every round.
		// Count live entries by ranging over the lru cache.
		live := countLiveInMemory(idx)
		t.Logf("InMemory no-sweep: %d live entries after %d rounds (%d added per round)",
			live, rounds, keysPerRound)
		assert.GreaterOrEqual(t, live, rounds*keysPerRound,
			"expected stale entries to accumulate without Sweep")
	})

	t.Run("InMemory_ExplicitSweep_BoundedGrowth", func(t *testing.T) {
		idx, err := NewInMemoryIndex(nil)
		require.NoError(t, err)

		for round := 0; round < rounds; round++ {
			keys := keysForRound(round, keysPerRound)
			require.NoError(t, idx.Add(t.Context(), keys, keys, []PodEntry{podEntry}))
			require.NoError(t, idx.Clear(t.Context(), podEntry))
			removed := idx.Sweep(t.Context())
			assert.Equal(t, keysPerRound, removed,
				"round %d: Sweep should remove this round's stale entries", round)
		}

		// After a final round + sweep, the index should be empty.
		live := countLiveInMemory(idx)
		t.Logf("InMemory explicit-sweep: %d live entries after %d rounds", live, rounds)
		assert.Equal(t, 0, live, "Sweep should keep growth bounded")
	})

	t.Run("InMemory_BackgroundSweeper_BoundedGrowth", func(t *testing.T) {
		idx, err := NewInMemoryIndex(nil)
		require.NoError(t, err)

		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		go idx.StartSweeper(ctx, 20*time.Millisecond)

		for round := 0; round < rounds; round++ {
			keys := keysForRound(round, keysPerRound)
			require.NoError(t, idx.Add(t.Context(), keys, keys, []PodEntry{podEntry}))
			require.NoError(t, idx.Clear(t.Context(), podEntry))
			// Give the sweeper a chance to run.
			time.Sleep(40 * time.Millisecond)
		}

		// Wait one more debounce window to drain.
		time.Sleep(60 * time.Millisecond)

		live := countLiveInMemory(idx)
		t.Logf("InMemory background-sweeper: %d live entries after %d rounds", live, rounds)
		// Bound: allow up to a small constant of in-flight entries from the most recent round.
		assert.LessOrEqual(t, live, keysPerRound,
			"background sweeper should keep buildup bounded near a round's worth")
	})

	t.Run("CostAware_ExplicitSweep_BoundedGrowth", func(t *testing.T) {
		idx, err := NewCostAwareMemoryIndex(nil)
		require.NoError(t, err)

		for round := 0; round < rounds; round++ {
			keys := keysForRound(round, keysPerRound)
			require.NoError(t, idx.Add(t.Context(), keys, keys, []PodEntry{podEntry}))
			require.NoError(t, idx.Clear(t.Context(), podEntry))
			removed := idx.Sweep(t.Context())
			assert.Equal(t, keysPerRound, removed,
				"round %d: Sweep should remove this round's stale entries", round)
		}
	})

	t.Run("Redis_ExplicitSweep_BoundedGrowth", func(t *testing.T) {
		idx, addr, cleanup := newMiniRedisIndex(t)
		defer cleanup()
		_ = addr

		for round := 0; round < rounds; round++ {
			keys := keysForRound(round, keysPerRound)
			require.NoError(t, idx.Add(t.Context(), keys, keys, []PodEntry{podEntry}))
			require.NoError(t, idx.Clear(t.Context(), podEntry))
			removed, err := idx.Sweep(t.Context())
			require.NoError(t, err)
			assert.Equal(t, keysPerRound, removed,
				"round %d: Redis Sweep should remove this round's stale entries", round)
		}

		// After a final sweep, the request-key namespace should be empty.
		live := countLiveRedisRequestHashes(t.Context(), idx)
		t.Logf("Redis explicit-sweep: %d live request-key hashes after %d rounds", live, rounds)
		assert.Equal(t, 0, live, "Redis Sweep should keep growth bounded")
	})
}

// keysForRound generates a deterministic, round-unique set of BlockHash keys.
func keysForRound(round, n int) []BlockHash {
	r := rand.New(rand.NewPCG(uint64(round)+1, 0xC0FFEE))
	out := make([]BlockHash, n)
	for i := range out {
		out[i] = BlockHash(r.Uint64())
	}
	return out
}

// countLiveInMemory counts physically-present (PodEntry,requestKey) pairs in an
// InMemoryIndex's underlying LRU. Used by the build-up test to distinguish
// "Lookup reports zero" (correctness) from "the index has reclaimed memory".
func countLiveInMemory(idx *InMemoryIndex) int {
	n := 0
	for _, k := range idx.data.Keys() {
		pc, ok := idx.data.Peek(k)
		if !ok || pc == nil {
			continue
		}
		pc.mu.Lock()
		n += pc.cache.Len()
		pc.mu.Unlock()
	}
	return n
}

// newMiniRedisIndex spins up an in-process Redis (miniredis) and returns a
// RedisIndex pointed at it, plus a cleanup func.
func newMiniRedisIndex(t *testing.T) (*RedisIndex, string, func()) {
	t.Helper()
	mr, err := miniredis.Run()
	require.NoError(t, err)
	cfg := DefaultRedisIndexConfig()
	cfg.Address = mr.Addr()
	idx, err := NewRedisIndex(cfg)
	require.NoError(t, err)
	ri, ok := idx.(*RedisIndex)
	require.True(t, ok)
	return ri, mr.Addr(), func() {
		_ = ri.RedisClient.Close()
		mr.Close()
	}
}

// countLiveRedisRequestHashes scans all keys and counts request-key hashes
// (i.e. those without the "engine:" prefix) that still hold any fields.
func countLiveRedisRequestHashes(ctx context.Context, idx *RedisIndex) int {
	n := 0
	var cursor uint64
	for {
		keys, next, err := idx.RedisClient.Scan(ctx, cursor, "*", 256).Result()
		if err != nil {
			return n
		}
		for _, k := range keys {
			if hasPrefix(k, "engine:") {
				continue
			}
			if length, err := idx.RedisClient.HLen(ctx, k).Result(); err == nil && length > 0 {
				n++
			}
		}
		cursor = next
		if cursor == 0 {
			break
		}
	}
	return n
}

func hasPrefix(s, p string) bool {
	return len(s) >= len(p) && s[:len(p)] == p
}

