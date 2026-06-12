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
	"strings"
	"testing"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The HMA scoring tests run over 3 request keys at canonical block size 16.
// The default window of 32 tokens needs cdiv(31,16) = 2 contiguous trailing
// blocks for a sliding-window hit. Case names carry the vLLM semantic each
// scenario pins.
const hmaTestBlockSize = 16

type groupMetas = map[kvblock.GroupID]kvblock.GroupMetadata

func swaWindow(windowTokens int) *int { return &windowTokens }

// hmaCatalog models a full-attention (group 0) + sliding-window (group 1,
// window 32) hybrid.
func hmaCatalog(pods ...string) *kvblock.GroupCatalog {
	c := kvblock.NewGroupCatalog()
	for _, pod := range pods {
		c.Learn(pod, 0, kvblock.GroupMetadata{IsMainAttention: true, BlockSize: 16})
		c.Learn(pod, 1, kvblock.GroupMetadata{BlockSize: 16, SlidingWindowSize: swaWindow(32)})
	}
	return c
}

// catalogFor builds a catalog for podA from explicit group metadata.
func catalogFor(metas groupMetas) *kvblock.GroupCatalog {
	c := kvblock.NewGroupCatalog()
	for groupID, meta := range metas {
		c.Learn(podA, groupID, meta)
	}
	return c
}

func hmaEntry(pod string, groupIdx int) kvblock.PodEntry {
	return kvblock.PodEntry{PodIdentifier: pod, DeviceTier: "gpu", HasGroup: true, GroupIdx: kvblock.GroupID(groupIdx)}
}

// blocksFor maps keys 1..len(specs) to podA entries. Each spec lists the
// entries present at that block, space-separated: "1" = group 1 on gpu,
// "c1" = group 1 on cpu, "x" = an ungrouped legacy entry, "" = nothing cached.
func blocksFor(specs ...string) map[kvblock.BlockHash][]kvblock.PodEntry {
	keyToPods := make(map[kvblock.BlockHash][]kvblock.PodEntry, len(specs))
	for i, spec := range specs {
		var entries []kvblock.PodEntry
		for _, token := range strings.Fields(spec) {
			entry := kvblock.PodEntry{PodIdentifier: podA, DeviceTier: "gpu"}
			if token != "x" {
				if rest, isCPU := strings.CutPrefix(token, "c"); isCPU {
					entry.DeviceTier = "cpu"
					token = rest
				}
				entry.HasGroup = true
				entry.GroupIdx = kvblock.GroupID(token[0] - '0')
			}
			entries = append(entries, entry)
		}
		if entries != nil {
			keyToPods[kvblock.BlockHash(i+1)] = entries // #nosec G115 -- test data, i is small
		}
	}
	return keyToPods
}

func hmaScorer(catalog *kvblock.GroupCatalog) *kvcache.LongestPrefixScorer {
	return &kvcache.LongestPrefixScorer{
		MediumWeights:      map[string]float64{"gpu": 1.0, "cpu": 0.5},
		Catalog:            catalog,
		CanonicalBlockSize: hmaTestBlockSize,
	}
}

func assertHMAScore(t *testing.T, catalog *kvblock.GroupCatalog,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry, want float64,
) {
	t.Helper()
	keys := int64KeysToKVBlockKeys([]uint64{1, 2, 3})
	scores, err := hmaScorer(catalog).Score(context.Background(), keys, keyToPods)
	assert.NoError(t, err)
	assert.InDelta(t, want, scores[podA], 0.0001)
}

// TestHMAScoring verifies HMA window-aware scoring: the main-attention group
// (group 0) gates the contiguous prefix and the sliding-window group (group 1)
// reduces it to the longest prefix whose trailing window is present, mirroring
// vLLM's SlidingWindowManager.find_longest_cache_hit (only the trailing window
// is required, early blocks are not; a block-0-anchored run shorter than the
// window still hits).
func TestHMAScoring(t *testing.T) {
	base := hmaCatalog(podA)
	window64 := catalogFor(groupMetas{0: {IsMainAttention: true}, 1: {SlidingWindowSize: swaWindow(64)}})
	engineBS64 := catalogFor(groupMetas{
		0: {IsMainAttention: true, BlockSize: 64},
		1: {BlockSize: 64, SlidingWindowSize: swaWindow(32)},
	})
	swaLearnedOnly := catalogFor(groupMetas{1: {SlidingWindowSize: swaWindow(32)}})

	tests := []struct {
		name    string
		catalog *kvblock.GroupCatalog
		blocks  map[kvblock.BlockHash][]kvblock.PodEntry
		want    float64
	}{
		{"full hit - main and SWA at every block", base, blocksFor("0 1", "0 1", "0 1"), 3.0},
		{"SWA group absent for prefix - converged hit collapses", base, blocksFor("0", "0", "0"), 0.0},
		{"SWA trailing window evicted - hit shrinks to rightmost window", base, blocksFor("0 1", "0 1", "0"), 2.0},
		{"early SWA block evicted - only the trailing window is required", base, blocksFor("0", "0 1", "0 1"), 3.0},
		{"SWA window never completes - block-0-anchored run still hits", base, blocksFor("0 1", "0", "0"), 1.0},
		{"window larger than prefix - full prefix still hits", window64, blocksFor("0 1", "0 1", "0 1"), 3.0},
		{"non-main entry at block 0 does not anchor the prefix", base, blocksFor("1", "0 1", "0 1"), 0.0},
		{"hybrid pod, main cold, SWA warm - fallback must not claim it", base, blocksFor("1", "1", "1"), 0.0},
		{"chain breaks where the main group goes missing", base, blocksFor("0 1", "0 1", "1"), 2.0},
		{"window divides by canonical block size, not engine's (64)", engineBS64, blocksFor("0", "0", "0 1"), 0.0},
		{"mixed tiers with truncation - weights sum over the hit only", base, blocksFor("0 1", "c0 1", "0"), 1.5},
		{"group 0 unlearned counts as main, learned SWA still gates", swaLearnedOnly, blocksFor("0 1", "0 1", "0"), 2.0},
		{"non-HMA entries - legacy behavior unchanged", kvblock.NewGroupCatalog(), blocksFor("x", "x", "x"), 3.0},
		{"empty catalog - group_idx 0 fallback treats group 0 as main", kvblock.NewGroupCatalog(), blocksFor("0 1", "0 1", "0 1"), 3.0},
		{"nil catalog - group_idx 0 fallback still routes on group 0", nil, blocksFor("0", "0", "1"), 2.0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) { assertHMAScore(t, tt.catalog, tt.blocks, tt.want) })
	}
}

// TestHMAScoring_TwoPods verifies chain independence between pods.
func TestHMAScoring_TwoPods(t *testing.T) {
	keys := int64KeysToKVBlockKeys([]uint64{1, 2, 3})
	keyToPods := map[kvblock.BlockHash][]kvblock.PodEntry{
		1: {hmaEntry(podA, 0), hmaEntry(podA, 1), hmaEntry(podB, 0), hmaEntry(podB, 1)},
		2: {hmaEntry(podA, 0), hmaEntry(podA, 1)}, // podB drops here
		3: {hmaEntry(podA, 0), hmaEntry(podA, 1)},
	}

	scores, err := hmaScorer(hmaCatalog(podA, podB)).Score(context.Background(), keys, keyToPods)
	assert.NoError(t, err)
	assert.InDelta(t, 3.0, scores[podA], 0.0001)
	assert.InDelta(t, 1.0, scores[podB], 0.0001)
}

// TestHMAScoring_MultiSWAGroups verifies that same-window SWA groups are
// scanned jointly with AND-presence, mirroring vLLM's per-spec-group lookup
// (a miss in any group is a miss). Per-group scans with a min would overstate:
// in the first case each group alone has a 2-block run (group 1 at blocks
// 1..2, group 2 at 0..1) but no jointly-present window exists.
func TestHMAScoring_MultiSWAGroups(t *testing.T) {
	catalog := catalogFor(groupMetas{
		0: {IsMainAttention: true},
		1: {SlidingWindowSize: swaWindow(32)},
		2: {SlidingWindowSize: swaWindow(32)},
	})
	assertHMAScore(t, catalog, blocksFor("0 2", "0 1 2", "0 1"), 0.0) // disjoint windows - no joint hit
	assertHMAScore(t, catalog, blocksFor("0", "0 1 2", "0 1 2"), 3.0) // shared trailing window - full hit
}

// TestHMAScoring_HeterogeneousWindows verifies fixed-point convergence across
// window classes, mirroring vLLM's restart-on-shrink loop (SWA hits are not
// downward-closed): group 2 (window 17, need 1) shrinks the hit to 1, which
// invalidates group 1's window (nothing at block 0) - the fixed point is 0,
// where a single sequential pass would have stopped at 1.
func TestHMAScoring_HeterogeneousWindows(t *testing.T) {
	catalog := catalogFor(groupMetas{
		0: {IsMainAttention: true},
		1: {SlidingWindowSize: swaWindow(32)},
		2: {SlidingWindowSize: swaWindow(17)},
	})
	assertHMAScore(t, catalog, blocksFor("0 2", "0 1", "0 1"), 0.0)
}

// TestHMAScoring_SWAOnlyModel verifies the unitary-coordinator mirror for
// models with no main-attention group: the trailing-window scan runs over the
// whole key range, cached blocks score at their tier weight, and null-prefix
// blocks (outside the window, never cached, skipped by the engine entirely)
// count at the tier-independent weight 1.0.
func TestHMAScoring_SWAOnlyModel(t *testing.T) {
	catalog := catalogFor(groupMetas{0: {SlidingWindowSize: swaWindow(32)}})
	assertHMAScore(t, catalog, blocksFor("", "0", "0"), 3.0)   // null prefix (1.0) + window
	assertHMAScore(t, catalog, blocksFor("", "c0", "c0"), 2.0) // cpu run 0.5+0.5, null prefix 1.0
	assertHMAScore(t, catalog, blocksFor("0", "", ""), 1.0)    // window never completes - anchored run
	assertHMAScore(t, catalog, blocksFor("", "", ""), 0.0)     // nothing cached
}

// TestHMAScoring_IndexerWiring exercises the real construction path:
// NewKVCacheIndexer must wire the token processor's block size into the scorer
// and SetGroupCatalog must reach it. Every other test sets those fields
// directly, so only this test catches a broken wire - an unwired scorer
// reports a full hit (3.0) where the catalog demands a collapse (0.0).
func TestHMAScoring_IndexerWiring(t *testing.T) {
	ctx := context.Background()

	config, err := kvcache.NewDefaultConfig()
	require.NoError(t, err)
	tp, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{
		BlockSizeTokens: hmaTestBlockSize,
		HashSeed:        "test",
	})
	require.NoError(t, err)
	indexer, err := kvcache.NewKVCacheIndexer(ctx, config, tp)
	require.NoError(t, err)
	indexer.SetGroupCatalog(hmaCatalog(podA))

	tokens := make([]uint32, 3*hmaTestBlockSize)
	for i := range tokens {
		tokens[i] = uint32(i + 1) // #nosec G115 -- test data, i is small
	}
	keys, err := tp.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "test-model", nil)
	require.NoError(t, err)
	require.Len(t, keys, 3)

	addEntries := func(groupIdx int) {
		for _, key := range keys {
			require.NoError(t, indexer.KVBlockIndex().Add(ctx,
				[]kvblock.BlockHash{key}, []kvblock.BlockHash{key},
				[]kvblock.PodEntry{hmaEntry(podA, groupIdx)}))
		}
	}

	// Main-attention entries only: SWA absent, a wired scorer collapses to 0.
	addEntries(0)
	scores, err := indexer.ScoreTokens(ctx, tokens, "test-model", nil, nil)
	require.NoError(t, err)
	assert.InDelta(t, 0.0, scores[podA], 0.0001)

	// Positive control: with the SWA group present the full hit returns.
	addEntries(1)
	scores, err = indexer.ScoreTokens(ctx, tokens, "test-model", nil, nil)
	require.NoError(t, err)
	assert.InDelta(t, 3.0, scores[podA], 0.0001)
}
