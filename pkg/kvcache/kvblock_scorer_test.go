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

	scored, err := scorer.Score(context.Background(), blockKeys, hitmap, testModelName)
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

	scored, err := scorer.Score(context.Background(), blockKeys, hitmap, testModelName)
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

// TestHybridPrefixCacheScorer tests the HybridPrefixMatch scorer using single-pass
// boundary evaluation: full attention is a kill switch, SWA tracks contiguous counts
// with sticky last_seq, and final score = min(all checkpoints) + 1.
func TestHybridPrefixCacheScorer(t *testing.T) {
	tests := []struct {
		name           string
		modelConfig    *kvcache.ModelConfig
		keys           []kvblock.BlockHash
		keyToPods      map[kvblock.BlockHash][]kvblock.PodEntry
		expectedScores map[string]float64
	}{
		{
			name: "FullAttentionOnly_FallsBackToLongestPrefix",
			modelConfig: &kvcache.ModelConfig{
				Name:  "TestModel",
				IsHMA: true,
				AttentionGroups: []kvcache.AttentionGroupConfig{
					{GroupID: 0, AttentionType: kvcache.AttentionTypeFull, BlockSize: 64},
				},
			},
			keys: []kvblock.BlockHash{100, 101, 102},
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				100: {
					{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: 1 << 0},
					{PodIdentifier: "podB", DeviceTier: "gpu", StoredGroups: 1 << 0},
				},
				101: {
					{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: 1 << 0},
				},
				102: {
					{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: 1 << 0},
				},
			},
			expectedScores: map[string]float64{
				"podA": 3.0,
				"podB": 1.0,
			},
		},
		{
			name: "SWAOnly_FallsBackToLongestPrefix",
			modelConfig: &kvcache.ModelConfig{
				Name:  "TestModel",
				IsHMA: true,
				AttentionGroups: []kvcache.AttentionGroupConfig{
					{GroupID: 1, AttentionType: kvcache.AttentionTypeSlidingWindow, BlockSize: 64, SlidingWindowSize: 129},
				},
			},
			keys: []kvblock.BlockHash{100, 101},
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				100: {{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: 1 << 1}},
				101: {{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: 1 << 1}},
			},
			expectedScores: map[string]float64{
				"podA": 2.0,
			},
		},
		{
			name: "HybridModel_FullAndSWA",
			modelConfig: &kvcache.ModelConfig{
				Name:  "DeepSeek-V3",
				IsHMA: true,
				AttentionGroups: []kvcache.AttentionGroupConfig{
					{GroupID: 0, AttentionType: kvcache.AttentionTypeFull, BlockSize: 64},
					{GroupID: 1, AttentionType: kvcache.AttentionTypeSlidingWindow, BlockSize: 64, SlidingWindowSize: 129},
				},
			},
			keys: []kvblock.BlockHash{100, 101, 102},
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				100: {
					{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: (1 << 0) | (1 << 1)},
					{PodIdentifier: "podB", DeviceTier: "gpu", StoredGroups: (1 << 0) | (1 << 1)},
				},
				101: {
					{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: (1 << 0) | (1 << 1)},
					{PodIdentifier: "podB", DeviceTier: "gpu", StoredGroups: (1 << 0) | (1 << 1)},
				},
				102: {
					{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: (1 << 0) | (1 << 1)},
				},
			},
			expectedScores: map[string]float64{
				"podA": 3.0, // full:0-2, swa count reaches threshold at b=1 and b=2, checkpoint=min(2,2)=2
				"podB": 2.0, // full:0-1 (break at 102), swa threshold met at b=1, checkpoint=min(1,1)=1
			},
		},
		{
			name: "NoCandidateWithoutFullGroupAtBlock0",
			modelConfig: &kvcache.ModelConfig{
				Name:  "TestModel",
				IsHMA: true,
				AttentionGroups: []kvcache.AttentionGroupConfig{
					{GroupID: 0, AttentionType: kvcache.AttentionTypeFull, BlockSize: 64},
					{GroupID: 1, AttentionType: kvcache.AttentionTypeSlidingWindow, BlockSize: 64, SlidingWindowSize: 129},
				},
			},
			keys: []kvblock.BlockHash{100, 101},
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				100: {
					{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: (1 << 0) | (1 << 1)},
					{PodIdentifier: "podB", DeviceTier: "gpu", StoredGroups: 1 << 1},
				},
				101: {
					{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: (1 << 0) | (1 << 1)},
					{PodIdentifier: "podB", DeviceTier: "gpu", StoredGroups: 1 << 1},
				},
			},
			expectedScores: map[string]float64{
				"podA": 2.0,
			},
		},
		{
			name: "SWA_BelowThreshold_ZeroScore",
			modelConfig: &kvcache.ModelConfig{
				Name:  "TestModel",
				IsHMA: true,
				AttentionGroups: []kvcache.AttentionGroupConfig{
					{GroupID: 0, AttentionType: kvcache.AttentionTypeFull, BlockSize: 64},
					{GroupID: 1, AttentionType: kvcache.AttentionTypeSlidingWindow, BlockSize: 64, SlidingWindowSize: 193},
					// threshold = cdiv(192, 64) = 3, only 2 blocks available
				},
			},
			keys: []kvblock.BlockHash{100, 101},
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				100: {{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: (1 << 0) | (1 << 1)}},
				101: {{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: (1 << 0) | (1 << 1)}},
			},
			expectedScores: map[string]float64{},
		},
		{
			name: "SWA_GapAndRecovery_StickyLastSeq",
			modelConfig: &kvcache.ModelConfig{
				Name:  "TestModel",
				IsHMA: true,
				AttentionGroups: []kvcache.AttentionGroupConfig{
					{GroupID: 0, AttentionType: kvcache.AttentionTypeFull, BlockSize: 64},
					{GroupID: 1, AttentionType: kvcache.AttentionTypeSlidingWindow, BlockSize: 64, SlidingWindowSize: 129},
					// threshold = cdiv(128, 64) = 2
				},
			},
			keys: []kvblock.BlockHash{100, 101, 102, 103, 104},
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				100: {{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: (1 << 0) | (1 << 1)}},
				101: {{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: (1 << 0) | (1 << 1)}},
				102: {{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: 1 << 0}}, // SWA gap
				103: {{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: (1 << 0) | (1 << 1)}},
				104: {{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: (1 << 0) | (1 << 1)}},
			},
			expectedScores: map[string]float64{
				// swa: count reaches 2 at b=1 (last_seq=1), resets at b=2, reaches 2 again at b=4 (last_seq=4)
				// checkpoint = min(lastSeqFull=4, swaLastSeqs=4) = 4, score = 5
				"podA": 5.0,
			},
		},
		{
			name: "SWA_StickyCheckpoint_LimitsScore",
			modelConfig: &kvcache.ModelConfig{
				Name:  "TestModel",
				IsHMA: true,
				AttentionGroups: []kvcache.AttentionGroupConfig{
					{GroupID: 0, AttentionType: kvcache.AttentionTypeFull, BlockSize: 64},
					{GroupID: 1, AttentionType: kvcache.AttentionTypeSlidingWindow, BlockSize: 64, SlidingWindowSize: 129},
				},
			},
			keys: []kvblock.BlockHash{100, 101, 102, 103},
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				100: {{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: (1 << 0) | (1 << 1)}},
				101: {{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: (1 << 0) | (1 << 1)}},
				102: {{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: 1 << 0}}, // SWA gap
				103: {{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: (1 << 0) | (1 << 1)}},
			},
			expectedScores: map[string]float64{
				// swa: threshold met at b=1 (last_seq=1), gap at b=2, only 1 hit at b=3 (< threshold)
				// checkpoint = min(lastSeqFull=3, swaLastSeqs=1) = 1, score = 2
				"podA": 2.0,
			},
		},
		{
			name: "NonStandardGroupIDs",
			modelConfig: &kvcache.ModelConfig{
				Name:  "TestModel",
				IsHMA: true,
				AttentionGroups: []kvcache.AttentionGroupConfig{
					{GroupID: 5, AttentionType: kvcache.AttentionTypeFull, BlockSize: 64},
					{GroupID: 3, AttentionType: kvcache.AttentionTypeSlidingWindow, BlockSize: 64, SlidingWindowSize: 129},
				},
			},
			keys: []kvblock.BlockHash{100, 101, 102},
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				100: {{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: (1 << 5) | (1 << 3)}},
				101: {{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: (1 << 5) | (1 << 3)}},
				102: {{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: (1 << 5) | (1 << 3)}},
			},
			expectedScores: map[string]float64{
				"podA": 3.0,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := &kvcache.KVBlockScorerConfig{
				ModelConfigs: []*kvcache.ModelConfig{tt.modelConfig},
				BackendConfigs: []*kvcache.KVCacheBackendConfig{
					{Name: "gpu", Weight: 1.0},
				},
			}

			scorer, err := kvcache.NewKVBlockScorer(config)
			assert.NoError(t, err)
			assert.Equal(t, kvcache.HybridPrefixMatch, scorer.Strategy())

			ctx := context.Background()
			scores, err := scorer.Score(ctx, tt.keys, tt.keyToPods, tt.modelConfig.Name)
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

// TestScorerSelection tests automatic scorer selection based on model configuration.
func TestScorerSelection(t *testing.T) {
	tests := []struct {
		name             string
		modelConfigs     []*kvcache.ModelConfig
		expectedStrategy kvcache.KVScoringStrategy
	}{
		{
			name:             "NoModels_UseSimpleScorer",
			modelConfigs:     nil,
			expectedStrategy: kvcache.LongestPrefixMatch,
		},
		{
			name: "OnlySimpleModels_UseSimpleScorer",
			modelConfigs: []*kvcache.ModelConfig{
				{Name: "Qwen/Qwen3-8B", IsHMA: false},
				{Name: "Llama-3-8B", IsHMA: false},
			},
			expectedStrategy: kvcache.LongestPrefixMatch,
		},
		{
			name: "OnlyHMAModels_UseHybridScorer",
			modelConfigs: []*kvcache.ModelConfig{
				{
					Name:  "DeepSeek-V3",
					IsHMA: true,
					AttentionGroups: []kvcache.AttentionGroupConfig{
						{GroupID: 0, AttentionType: kvcache.AttentionTypeFull, BlockSize: 64},
						{
							GroupID:           1,
							AttentionType:     kvcache.AttentionTypeSlidingWindow,
							BlockSize:         64,
							SlidingWindowSize: 4096,
						},
					},
				},
			},
			expectedStrategy: kvcache.HybridPrefixMatch,
		},
		{
			name: "MixedModels_UseHybridScorer",
			modelConfigs: []*kvcache.ModelConfig{
				{Name: "Qwen/Qwen3-8B", IsHMA: false},
				{
					Name:  "DeepSeek-V3",
					IsHMA: true,
					AttentionGroups: []kvcache.AttentionGroupConfig{
						{GroupID: 0, AttentionType: kvcache.AttentionTypeFull, BlockSize: 64},
					},
				},
			},
			expectedStrategy: kvcache.HybridPrefixMatch,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			scorerConfig := kvcache.DefaultKVBlockScorerConfig()
			scorerConfig.ModelConfigs = tt.modelConfigs

			scorer, err := kvcache.NewKVBlockScorer(scorerConfig)
			assert.NoError(t, err)
			assert.Equal(t, tt.expectedStrategy, scorer.Strategy())
		})
	}
}
