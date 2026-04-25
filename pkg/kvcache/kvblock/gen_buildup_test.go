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

// TestGenBuildup runs N rounds of Add+Clear with fresh request keys per round
// and verifies the three contracts of the generation-counter approach:
//
//  1. Correctness without Sweep: Lookup returns zero hits post-Clear even though
//     entries physically linger in the index.
//  2. Default-on Sweeper via NewIndex: stale entries are reclaimed automatically.
//  3. Redis Sweep: explicit reclamation works against the Redis backend.
func TestGenBuildup(t *testing.T) {
	const (
		rounds       = 10
		keysPerRound = 500
	)
	podEntry := PodEntry{PodIdentifier: "podA", DeviceTier: "gpu"}

	t.Run("NoSweep_LookupCorrectButEntriesLinger", func(t *testing.T) {
		idx, err := NewInMemoryIndex(nil)
		require.NoError(t, err)

		for round := 0; round < rounds; round++ {
			keys := keysForRound(round, keysPerRound)
			require.NoError(t, idx.Add(t.Context(), keys, keys, []PodEntry{podEntry}))
			hits, _ := idx.Lookup(t.Context(), keys, sets.Set[string]{})
			assert.Len(t, hits, keysPerRound)
			require.NoError(t, idx.Clear(t.Context(), podEntry))
			postClear, _ := idx.Lookup(t.Context(), keys, sets.Set[string]{})
			assert.Empty(t, postClear, "Lookup must return zero post-Clear")
		}

		live := countLiveInMemory(idx)
		t.Logf("no-sweep: %d live entries after %d rounds (%d/round)", live, rounds, keysPerRound)
		assert.GreaterOrEqual(t, live, rounds*keysPerRound)
	})

	t.Run("NewIndex_DefaultSweeper_BoundedGrowth", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()

		cfg := DefaultIndexConfig()
		cfg.SweeperDebounce = 20 * time.Millisecond
		idx, err := NewIndex(ctx, cfg)
		require.NoError(t, err)

		for round := 0; round < rounds; round++ {
			keys := keysForRound(round, keysPerRound)
			require.NoError(t, idx.Add(ctx, keys, keys, []PodEntry{podEntry}))
			require.NoError(t, idx.Clear(ctx, podEntry))
			time.Sleep(40 * time.Millisecond)
		}
		time.Sleep(60 * time.Millisecond)

		live := countLiveInMemory(idx.(*InMemoryIndex))
		t.Logf("default sweeper: %d live entries after %d rounds", live, rounds)
		assert.LessOrEqual(t, live, keysPerRound)
	})

	t.Run("Redis_Sweep_BoundedGrowth", func(t *testing.T) {
		idx, _, cleanup := newMiniRedisIndex(t)
		defer cleanup()

		for round := 0; round < rounds; round++ {
			keys := keysForRound(round, keysPerRound)
			require.NoError(t, idx.Add(t.Context(), keys, keys, []PodEntry{podEntry}))
			require.NoError(t, idx.Clear(t.Context(), podEntry))
			removed, err := idx.Sweep(t.Context())
			require.NoError(t, err)
			assert.Equal(t, keysPerRound, removed)
		}
		assert.Equal(t, 0, countLiveRedisRequestHashes(t.Context(), idx))
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

