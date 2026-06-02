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

package kvcache_test

import (
	"context"
	"testing"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/stretchr/testify/assert"
)

const (
	testModelName = "test-model"
	podA          = "pod-a"
	podB          = "pod-b"
)

// TestLongestPrefixScorer verifies scoring based on consecutive block hits from the start.
func TestLongestPrefixScorer(t *testing.T) {
	mediumWeights := map[string]float64{
		"gpu": 1.0,
		"cpu": 0.5,
	}

	scorer := &kvcache.LongestPrefixScorer{
		MediumWeights: mediumWeights,
	}
	blockKeys := int64KeysToKVBlockKeys([]uint64{1001, 1002, 1003, 1004, 1005, 1006})

	hitmap := map[kvblock.BlockHash][]kvblock.PodEntry{
		1001: {{PodIdentifier: podA, DeviceTier: "gpu"}},
		1002: {{PodIdentifier: podA, DeviceTier: "gpu"}},
		1003: {
			{PodIdentifier: podA, DeviceTier: "gpu"},
			{PodIdentifier: podA, DeviceTier: "cpu"},
		},
		1004: {{PodIdentifier: podB, DeviceTier: "cpu"}},
		1005: {{PodIdentifier: podB, DeviceTier: "cpu"}},
		1006: {{PodIdentifier: podA, DeviceTier: "gpu"}},
	}

	expected := map[string]float64{
		podA: 3.0,
		podB: 0.0,
	}

	scored, err := scorer.Score(context.Background(), blockKeys, hitmap, nil)
	assert.NoError(t, err)
	for pod, score := range scored {
		assert.InDelta(t, expected[pod], score, 0.0001)
	}
}

func TestLongestPrefixScorerDifferentTiers(t *testing.T) {
	mediumWeights := map[string]float64{
		"gpu": 1.0,
		"cpu": 0.5,
	}

	scorer := &kvcache.LongestPrefixScorer{
		MediumWeights: mediumWeights,
	}
	blockKeys := int64KeysToKVBlockKeys([]uint64{1001, 1002, 1003, 1004, 1005, 1006})

	hitmap := map[kvblock.BlockHash][]kvblock.PodEntry{
		1001: {{PodIdentifier: podA, DeviceTier: "gpu"}},
		1002: {{PodIdentifier: podA, DeviceTier: "gpu"}},
		1003: {{PodIdentifier: podA, DeviceTier: "cpu"}},
		1004: {{PodIdentifier: podB, DeviceTier: "cpu"}},
		1005: {{PodIdentifier: podB, DeviceTier: "cpu"}},
		1006: {{PodIdentifier: podA, DeviceTier: "gpu"}},
	}

	expected := map[string]float64{
		podA: 2.5,
		podB: 0.0,
	}

	scored, err := scorer.Score(context.Background(), blockKeys, hitmap, nil)
	assert.NoError(t, err)
	for pod, score := range scored {
		assert.InDelta(t, expected[pod], score, 0.0001)
	}
}

func int64KeysToKVBlockKeys(keys []uint64) []kvblock.BlockHash {
	kvKeys := make([]kvblock.BlockHash, len(keys))
	for i, key := range keys {
		kvKeys[i] = kvblock.BlockHash(key)
	}
	return kvKeys
}

// attInfo creates a standard full+SWA AttentionInfo for tests.
func attInfo(fullGroupID int, swaGroupIDs, swaWindowBlocks []int) *kvblock.AttentionInfo {
	return &kvblock.AttentionInfo{
		FullGroupID:     fullGroupID,
		SWAGroupIDs:     swaGroupIDs,
		SWAWindowBlocks: swaWindowBlocks,
	}
}

// pe is a helper to build a PodEntry with a specific group.
func pe(podID, tier string, groupID int) kvblock.PodEntry { //nolint:unparam // tier kept as param for future mixed-tier tests
	return kvblock.PodEntry{PodIdentifier: podID, DeviceTier: tier, GroupID: groupID}
}

// TestHybridPrefixCacheScorer tests the HybridPrefixMatch scorer using single-pass
// boundary evaluation: full attention is a kill switch, SWA tracks contiguous counts
// with sticky last_seq, and final score = min(all checkpoints) + 1.
func TestHybridPrefixCacheScorer(t *testing.T) {
	tests := []struct {
		name           string
		attentionInfo  *kvblock.AttentionInfo // nil means fallback to LongestPrefix
		keys           []kvblock.BlockHash
		keyToPods      map[kvblock.BlockHash][]kvblock.PodEntry
		expectedScores map[string]float64
	}{
		{
			name:          "FullAttentionOnly_FallsBackToLongestPrefix",
			attentionInfo: nil,
			keys:          []kvblock.BlockHash{100, 101, 102},
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				100: {pe("podA", "gpu", 0), pe("podB", "gpu", 0)},
				101: {pe("podA", "gpu", 0)},
				102: {pe("podA", "gpu", 0)},
			},
			expectedScores: map[string]float64{
				"podA": 3.0,
				"podB": 1.0,
			},
		},
		{
			name:          "SWAOnly_FallsBackToLongestPrefix",
			attentionInfo: nil,
			keys:          []kvblock.BlockHash{100, 101},
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				100: {pe("podA", "gpu", 1)},
				101: {pe("podA", "gpu", 1)},
			},
			expectedScores: map[string]float64{
				"podA": 2.0,
			},
		},
		{
			name:          "HybridModel_FullAndSWA",
			attentionInfo: attInfo(0, []int{1}, []int{2}), // threshold=2
			keys:          []kvblock.BlockHash{100, 101, 102},
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				100: {pe("podA", "gpu", 0), pe("podA", "gpu", 1), pe("podB", "gpu", 0), pe("podB", "gpu", 1)},
				101: {pe("podA", "gpu", 0), pe("podA", "gpu", 1), pe("podB", "gpu", 0), pe("podB", "gpu", 1)},
				102: {pe("podA", "gpu", 0), pe("podA", "gpu", 1)},
			},
			expectedScores: map[string]float64{
				"podA": 3.0,
				"podB": 2.0,
			},
		},
		{
			name:          "NoCandidateWithoutFullGroupAtBlock0",
			attentionInfo: attInfo(0, []int{1}, []int{2}),
			keys:          []kvblock.BlockHash{100, 101},
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				100: {pe("podA", "gpu", 0), pe("podA", "gpu", 1), pe("podB", "gpu", 1)},
				101: {pe("podA", "gpu", 0), pe("podA", "gpu", 1), pe("podB", "gpu", 1)},
			},
			expectedScores: map[string]float64{
				"podA": 2.0,
			},
		},
		{
			name:          "SWA_BelowThreshold_ZeroScore",
			attentionInfo: attInfo(0, []int{1}, []int{3}), // threshold=3, only 2 blocks
			keys:          []kvblock.BlockHash{100, 101},
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				100: {pe("podA", "gpu", 0), pe("podA", "gpu", 1)},
				101: {pe("podA", "gpu", 0), pe("podA", "gpu", 1)},
			},
			expectedScores: map[string]float64{},
		},
		{
			name:          "SWA_GapAndRecovery_StickyLastSeq",
			attentionInfo: attInfo(0, []int{1}, []int{2}), // threshold=2
			keys:          []kvblock.BlockHash{100, 101, 102, 103, 104},
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				100: {pe("podA", "gpu", 0), pe("podA", "gpu", 1)},
				101: {pe("podA", "gpu", 0), pe("podA", "gpu", 1)},
				102: {pe("podA", "gpu", 0)}, // SWA gap
				103: {pe("podA", "gpu", 0), pe("podA", "gpu", 1)},
				104: {pe("podA", "gpu", 0), pe("podA", "gpu", 1)},
			},
			expectedScores: map[string]float64{
				"podA": 5.0,
			},
		},
		{
			name:          "SWA_StickyCheckpoint_LimitsScore",
			attentionInfo: attInfo(0, []int{1}, []int{2}),
			keys:          []kvblock.BlockHash{100, 101, 102, 103},
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				100: {pe("podA", "gpu", 0), pe("podA", "gpu", 1)},
				101: {pe("podA", "gpu", 0), pe("podA", "gpu", 1)},
				102: {pe("podA", "gpu", 0)}, // SWA gap
				103: {pe("podA", "gpu", 0), pe("podA", "gpu", 1)},
			},
			expectedScores: map[string]float64{
				"podA": 2.0,
			},
		},
		{
			name:          "NonStandardGroupIDs",
			attentionInfo: attInfo(5, []int{3}, []int{2}),
			keys:          []kvblock.BlockHash{100, 101, 102},
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				100: {pe("podA", "gpu", 5), pe("podA", "gpu", 3)},
				101: {pe("podA", "gpu", 5), pe("podA", "gpu", 3)},
				102: {pe("podA", "gpu", 5), pe("podA", "gpu", 3)},
			},
			expectedScores: map[string]float64{
				"podA": 3.0,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := &kvcache.KVBlockScorerConfig{
				BackendConfigs: []*kvcache.KVCacheBackendConfig{
					{Name: "gpu", Weight: 1.0},
				},
			}

			scorer, err := kvcache.NewKVBlockScorer(config)
			assert.NoError(t, err)
			assert.Equal(t, kvcache.HybridPrefixMatch, scorer.Strategy())

			ctx := context.Background()
			scores, err := scorer.Score(ctx, tt.keys, tt.keyToPods, tt.attentionInfo)
			assert.NoError(t, err)

			assert.Equal(t, len(tt.expectedScores), len(scores),
				"score map length mismatch for test case: %s", tt.name)
			for pod, expected := range tt.expectedScores {
				assert.InDelta(t, expected, scores[pod], 0.0001,
					"score mismatch for pod %s in test case: %s", pod, tt.name)
			}
		})
	}
}

// TestScorerSelection tests scorer selection based on configuration.
func TestScorerSelection(t *testing.T) {
	tests := []struct {
		name             string
		scoringStrategy  kvcache.KVScoringStrategy
		expectedStrategy kvcache.KVScoringStrategy
	}{
		{
			name:             "Default_UsesHybridScorer",
			scoringStrategy:  "",
			expectedStrategy: kvcache.HybridPrefixMatch,
		},
		{
			name:             "ExplicitLongestPrefix",
			scoringStrategy:  kvcache.LongestPrefixMatch,
			expectedStrategy: kvcache.LongestPrefixMatch,
		},
		{
			name:             "ExplicitHybridPrefix",
			scoringStrategy:  kvcache.HybridPrefixMatch,
			expectedStrategy: kvcache.HybridPrefixMatch,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			scorerConfig := kvcache.DefaultKVBlockScorerConfig()
			scorerConfig.ScoringStrategy = tt.scoringStrategy

			scorer, err := kvcache.NewKVBlockScorer(scorerConfig)
			assert.NoError(t, err)
			assert.Equal(t, tt.expectedStrategy, scorer.Strategy())
		})
	}
}
