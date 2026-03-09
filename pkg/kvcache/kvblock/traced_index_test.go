// Copyright 2025 The llm-d Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package kvblock_test

import (
	"context"
	"testing"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/sets"
)

func TestNewTracedIndex(t *testing.T) {
	// Create a base in-memory index
	baseIdx, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
	require.NoError(t, err)

	// Wrap it with tracing
	tracedIdx := kvblock.NewTracedIndex(baseIdx)
	require.NotNil(t, tracedIdx)
}

func TestTracedIndexBehavior(t *testing.T) {
	ctx := context.Background()

	// Create a base in-memory index
	baseIdx, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
	require.NoError(t, err)

	// Wrap it with tracing
	tracedIdx := kvblock.NewTracedIndex(baseIdx)

	// Test Add operation
	engineKey := kvblock.BlockHash(123)
	requestKey := kvblock.BlockHash(789)
	entries := []kvblock.PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu", DataParallelRank: kvblock.NoDataParallelRank},
		{PodIdentifier: "pod2", DeviceTier: "cpu", DataParallelRank: kvblock.NoDataParallelRank},
	}

	err = tracedIdx.Add(ctx, []kvblock.BlockHash{engineKey}, []kvblock.BlockHash{requestKey}, entries)
	require.NoError(t, err)

	// Test Lookup operation with tracing
	result, err := tracedIdx.Lookup(ctx, []kvblock.BlockHash{requestKey}, sets.Set[string]{})
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Len(t, result[requestKey], 2)

	// Test Evict operation
	err = tracedIdx.Evict(ctx, engineKey, []kvblock.PodEntry{entries[0]})
	require.NoError(t, err)

	// Verify eviction worked (pod1 should be removed, pod2 should remain)
	result, err = tracedIdx.Lookup(ctx, []kvblock.BlockHash{requestKey}, sets.Set[string]{})
	require.NoError(t, err)
	require.Len(t, result[requestKey], 1)
	require.Equal(t, "pod2", result[requestKey][0].PodIdentifier)
}

func TestTracedIndexCacheHitMetrics(t *testing.T) {
	ctx := context.Background()

	baseIdx, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
	require.NoError(t, err)

	tracedIdx := kvblock.NewTracedIndex(baseIdx)

	// Add some data
	engineKeys := []kvblock.BlockHash{kvblock.BlockHash(1)}
	requestKeys := []kvblock.BlockHash{kvblock.BlockHash(2)}
	entries := []kvblock.PodEntry{{PodIdentifier: "pod1", DeviceTier: "gpu", DataParallelRank: kvblock.NoDataParallelRank}}

	err = tracedIdx.Add(ctx, engineKeys, requestKeys, entries)
	require.NoError(t, err)

	// Lookup should succeed and record cache hit
	result, err := tracedIdx.Lookup(ctx, requestKeys, sets.Set[string]{})
	require.NoError(t, err)
	require.Len(t, result[requestKeys[0]], 1)

	// Lookup non-existent key should record cache miss
	nonExistentKeys := []kvblock.BlockHash{kvblock.BlockHash(999)}
	result, err = tracedIdx.Lookup(ctx, nonExistentKeys, sets.Set[string]{})
	require.NoError(t, err)
	require.Len(t, result[nonExistentKeys[0]], 0)
}
