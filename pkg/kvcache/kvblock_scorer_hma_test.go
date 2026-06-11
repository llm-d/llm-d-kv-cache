/*
Copyright 2026 The llm-d Authors.

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

package kvcache_test

import (
	"context"
	"testing"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/stretchr/testify/assert"
)

// hmaCatalog builds a catalog modelling a full-attention + sliding-window
// hybrid: group 0 is the main (full attention) group, group 1 is sliding
// window. The window (32 tokens at block size 16) requires need =
// cdiv(32-1, 16) = 2 contiguous trailing blocks for a sliding-window hit.
func hmaCatalog(pods ...string) *kvblock.GroupCatalog {
	c := kvblock.NewGroupCatalog()
	window := 32
	for _, pod := range pods {
		c.Learn(pod, 0, kvblock.GroupMetadata{IsMainAttention: true, BlockSize: 16})
		c.Learn(pod, 1, kvblock.GroupMetadata{BlockSize: 16, SlidingWindowSize: &window})
	}
	return c
}

func hmaEntry(pod string, groupIdx int) kvblock.PodEntry {
	return kvblock.PodEntry{PodIdentifier: pod, DeviceTier: "gpu", HasGroup: true, GroupIdx: kvblock.GroupID(groupIdx)}
}

func hmaScorer(catalog *kvblock.GroupCatalog) *kvcache.LongestPrefixScorer {
	return &kvcache.LongestPrefixScorer{
		MediumWeights: map[string]float64{"gpu": 1.0, "cpu": 0.5},
		Catalog:       catalog,
	}
}

// TestHMAScoring verifies HMA window-aware scoring: the main-attention group
// (group 0) gates the contiguous prefix, and the sliding-window group (group 1)
// reduces it to the longest prefix whose trailing window (need=2 blocks) is
// present. This mirrors vLLM's hybrid cache-hit convergence to the minimum
// across groups.
func TestHMAScoring(t *testing.T) {
	keys := int64KeysToKVBlockKeys([]uint64{1, 2, 3})

	tests := []struct {
		name      string
		keyToPods map[kvblock.BlockHash][]kvblock.PodEntry
		catalog   *kvblock.GroupCatalog
		want      float64 // expected score for podA
	}{
		{
			name:    "full hit — main group present at every block",
			catalog: hmaCatalog(podA),
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {hmaEntry(podA, 0), hmaEntry(podA, 1)},
				2: {hmaEntry(podA, 0), hmaEntry(podA, 1)},
				3: {hmaEntry(podA, 0), hmaEntry(podA, 1)},
			},
			want: 3.0,
		},
		{
			// The whole SWA group is gone for this prefix, so vLLM's convergence
			// to the per-group minimum yields no hit even though full attention
			// is fully cached.
			name:    "SWA group absent for prefix — converged hit collapses",
			catalog: hmaCatalog(podA),
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {hmaEntry(podA, 0)},
				2: {hmaEntry(podA, 0)},
				3: {hmaEntry(podA, 0)},
			},
			want: 0.0,
		},
		{
			// Full attention cached for all 3 blocks, but the SWA trailing block
			// (block 2) was evicted. vLLM scans SWA right-to-left and the hit
			// shrinks to the rightmost contiguous window (blocks 0..1).
			name:    "SWA trailing window evicted — hit shrinks below full prefix",
			catalog: hmaCatalog(podA),
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {hmaEntry(podA, 0), hmaEntry(podA, 1)},
				2: {hmaEntry(podA, 0), hmaEntry(podA, 1)},
				3: {hmaEntry(podA, 0)}, // SWA trailing block gone
			},
			want: 2.0,
		},
		{
			// The early SWA block (block 0) is outside the window and was
			// evicted, but the trailing window (blocks 1..2) is intact, so vLLM
			// still gets a full-length hit. Early SWA presence is not required.
			name:    "early SWA block evicted, trailing window intact — full hit",
			catalog: hmaCatalog(podA),
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {hmaEntry(podA, 0)}, // SWA early block gone (outside window)
				2: {hmaEntry(podA, 0), hmaEntry(podA, 1)},
				3: {hmaEntry(podA, 0), hmaEntry(podA, 1)},
			},
			want: 3.0,
		},
		{
			name:    "miss — only the SWA group present at the first block",
			catalog: hmaCatalog(podA),
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {hmaEntry(podA, 1)},
				2: {hmaEntry(podA, 0), hmaEntry(podA, 1)},
				3: {hmaEntry(podA, 0), hmaEntry(podA, 1)},
			},
			want: 0.0,
		},
		{
			name:    "chain breaks where the main group goes missing",
			catalog: hmaCatalog(podA),
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {hmaEntry(podA, 0), hmaEntry(podA, 1)},
				2: {hmaEntry(podA, 0), hmaEntry(podA, 1)},
				3: {hmaEntry(podA, 1)}, // main group absent -> prefix stops at block 2
			},
			want: 2.0,
		},
		{
			name:    "non-HMA entries — legacy behavior unchanged",
			catalog: kvblock.NewGroupCatalog(),
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {{PodIdentifier: podA, DeviceTier: "gpu"}},
				2: {{PodIdentifier: podA, DeviceTier: "gpu"}},
				3: {{PodIdentifier: podA, DeviceTier: "gpu"}},
			},
			want: 3.0,
		},
		{
			name:    "unlearned groups — group_idx 0 fallback treats group 0 as main",
			catalog: kvblock.NewGroupCatalog(), // empty: nothing learned yet
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {hmaEntry(podA, 0), hmaEntry(podA, 1)},
				2: {hmaEntry(podA, 0), hmaEntry(podA, 1)},
				3: {hmaEntry(podA, 0), hmaEntry(podA, 1)},
			},
			want: 3.0,
		},
		{
			name:    "nil catalog — group_idx 0 fallback still routes on group 0",
			catalog: nil,
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {hmaEntry(podA, 0)},
				2: {hmaEntry(podA, 0)},
				3: {hmaEntry(podA, 1)}, // non-zero group ignored under fallback
			},
			want: 2.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			scores, err := hmaScorer(tt.catalog).Score(context.Background(), keys, tt.keyToPods)
			assert.NoError(t, err)
			assert.InDelta(t, tt.want, scores[podA], 0.0001)
		})
	}
}

// TestHMAScoring_TwoPods verifies chain independence between pods, scored on
// the main-attention group only.
func TestHMAScoring_TwoPods(t *testing.T) {
	catalog := hmaCatalog(podA, podB)

	keys := int64KeysToKVBlockKeys([]uint64{1, 2, 3})
	keyToPods := map[kvblock.BlockHash][]kvblock.PodEntry{
		1: {
			hmaEntry(podA, 0), hmaEntry(podA, 1),
			hmaEntry(podB, 0), hmaEntry(podB, 1),
		},
		2: {hmaEntry(podA, 0), hmaEntry(podA, 1)}, // podB drops here
		3: {hmaEntry(podA, 0), hmaEntry(podA, 1)},
	}

	scores, err := hmaScorer(catalog).Score(context.Background(), keys, keyToPods)
	assert.NoError(t, err)
	assert.InDelta(t, 3.0, scores[podA], 0.0001)
	assert.InDelta(t, 1.0, scores[podB], 0.0001)
}
