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

	scored, err := scorer.Score(context.Background(), blockKeys, hitmap)
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

	scored, err := scorer.Score(context.Background(), blockKeys, hitmap)
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

func TestContainsGroup(t *testing.T) {
	tests := []struct {
		name    string
		groups  []int
		groupID int
		want    bool
	}{
		{
			name:    "group found",
			groups:  []int{0, 1, 2},
			groupID: 1,
			want:    true,
		},
		{
			name:    "group not found",
			groups:  []int{0, 1, 2},
			groupID: 5,
			want:    false,
		},
		{
			name:    "nil groups - backwards compatibility",
			groups:  nil,
			groupID: 0,
			want:    true,
		},
		{
			name:    "empty groups - backwards compatibility",
			groups:  []int{},
			groupID: 0,
			want:    true,
		},
		{
			name:    "first group",
			groups:  []int{0, 1, 2},
			groupID: 0,
			want:    true,
		},
		{
			name:    "last group",
			groups:  []int{0, 1, 2},
			groupID: 2,
			want:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := kvcache.ContainsGroup(tt.groups, tt.groupID)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestHybridPrefixCacheScorer_BasicScoring(t *testing.T) {
	mediumWeights := map[string]float64{
		"gpu": 2.0,
		"cpu": 1.0,
	}

	modelConfig := &kvcache.ModelConfig{
		Name:      "test-model",
		BlockSize: 16,
		AttentionGroups: []kvcache.AttentionGroupConfig{
			{WindowSize: 0, AttentionType: "full"},       // Group 0: Full Attention
			{WindowSize: 2048, AttentionType: "sliding"}, // Group 1: SWA-2048 (128 blocks)
		},
	}

	scorer := &kvcache.HybridPrefixCacheScorer{
		MediumWeights: mediumWeights,
		ModelConfig:   modelConfig,
	}

	// Request: 128 blocks (2048 tokens @ 16 tokens/block)
	blockKeys := make([]kvblock.BlockHash, 128)
	for i := range blockKeys {
		blockKeys[i] = kvblock.BlockHash(1000 + i)
	}

	// Cache: pod-a has all 128 blocks on GPU for both groups
	hitmap := make(map[kvblock.BlockHash][]kvblock.PodEntry)
	for i := 0; i < 128; i++ {
		hitmap[blockKeys[i]] = []kvblock.PodEntry{
			{PodIdentifier: podA, DeviceTier: "gpu", StoredGroups: []int{0, 1}},
		}
	}

	scored, err := scorer.Score(context.Background(), blockKeys, hitmap)
	assert.NoError(t, err)
	assert.Contains(t, scored, podA)

	// Expected with weighted combination:
	// Full (Group 0): 128 blocks × 2.0 = 256.0
	// SWA (Group 1): 128 blocks × 2.0 = 256.0 (window = 128 blocks = all blocks)
	// fullRatio = 256.0 / 128 = 2.0
	// swaRatio = 256.0 / 128 = 2.0
	// Normalized: score = 0.7 × (256/128) + 0.3 × (256/128) = 0.7×2.0 + 0.3×2.0 = 2.0
	// But wait, we normalize by dividing score by totalBlocks, so:
	// fullRatio = 256.0 / (128 × 2.0) gives us the hit ratio weighted by tier
	// Actually the formula is: 0.7 × (fullScore / maxPossibleScore) + 0.3 × (swaScore / maxPossibleScore)
	// where maxPossibleScore = totalBlocks (not weighted)
	// So: fullRatio = 256.0 / 128 = 2.0, swaRatio = 256.0 / 128 = 2.0
	// score = 0.7 × 2.0 + 0.3 × 2.0 = 2.0
	assert.InDelta(t, 2.0, scored[podA], 0.01)
}

func TestHybridPrefixCacheScorer_CompletenessScoring(t *testing.T) {
	mediumWeights := map[string]float64{
		"gpu": 2.0,
		"cpu": 1.0,
	}

	modelConfig := &kvcache.ModelConfig{
		Name:      "test-model",
		BlockSize: 16,
		AttentionGroups: []kvcache.AttentionGroupConfig{
			{WindowSize: 0, AttentionType: "full"}, // Only Full Attention group
		},
	}

	scorer := &kvcache.HybridPrefixCacheScorer{
		MediumWeights: mediumWeights,
		ModelConfig:   modelConfig,
	}

	tests := []struct {
		name      string
		numBlocks int
		wantScore float64
	}{
		{
			name:      "64 blocks all GPU",
			numBlocks: 64,
			// Full: 64 × 2.0 = 128, SWA: 0 (no SWA group)
			// score = 0.7 × (128/64) + 0.3 × (0/64) = 0.7 × 2.0 = 1.4
			wantScore: 1.4,
		},
		{
			name:      "128 blocks all GPU",
			numBlocks: 128,
			// score = 0.7 × (256/128) + 0.3 × 0 = 1.4
			wantScore: 1.4,
		},
		{
			name:      "200 blocks all GPU",
			numBlocks: 200,
			// score = 0.7 × (400/200) + 0.3 × 0 = 1.4
			wantScore: 1.4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			blockKeys := make([]kvblock.BlockHash, tt.numBlocks)
			hitmap := make(map[kvblock.BlockHash][]kvblock.PodEntry)

			for i := range blockKeys {
				blockKeys[i] = kvblock.BlockHash(1000 + i)
				hitmap[blockKeys[i]] = []kvblock.PodEntry{
					{PodIdentifier: podA, DeviceTier: "gpu", StoredGroups: []int{0}},
				}
			}

			scored, err := scorer.Score(context.Background(), blockKeys, hitmap)
			assert.NoError(t, err)
			assert.Contains(t, scored, podA)
			assert.InDelta(t, tt.wantScore, scored[podA], 0.01)
		})
	}
}

func TestHybridPrefixCacheScorer_MixedTiers(t *testing.T) {
	mediumWeights := map[string]float64{
		"gpu": 2.0,
		"cpu": 1.0,
	}

	modelConfig := &kvcache.ModelConfig{
		Name:      "test-model",
		BlockSize: 16,
		AttentionGroups: []kvcache.AttentionGroupConfig{
			{WindowSize: 0, AttentionType: "full"}, // Only Full Attention
		},
	}

	scorer := &kvcache.HybridPrefixCacheScorer{
		MediumWeights: mediumWeights,
		ModelConfig:   modelConfig,
	}

	// 128 blocks: first 64 on GPU, last 64 on CPU
	blockKeys := make([]kvblock.BlockHash, 128)
	hitmap := make(map[kvblock.BlockHash][]kvblock.PodEntry)

	for i := 0; i < 128; i++ {
		blockKeys[i] = kvblock.BlockHash(1000 + i)
		tier := "gpu"
		if i >= 64 {
			tier = "cpu"
		}
		hitmap[blockKeys[i]] = []kvblock.PodEntry{
			{PodIdentifier: podA, DeviceTier: tier, StoredGroups: []int{0}},
		}
	}

	scored, err := scorer.Score(context.Background(), blockKeys, hitmap)
	assert.NoError(t, err)
	assert.Contains(t, scored, podA)

	// Full: (64 GPU × 2.0) + (64 CPU × 1.0) = 128 + 64 = 192
	// SWA: 0 (no SWA group)
	// score = 0.7 × (192/128) + 0.3 × 0 = 0.7 × 1.5 = 1.05
	assert.InDelta(t, 1.05, scored[podA], 0.01)
}

func TestHybridPrefixCacheScorer_MultiplePods(t *testing.T) {
	mediumWeights := map[string]float64{
		"gpu": 2.0,
		"cpu": 1.0,
	}

	modelConfig := &kvcache.ModelConfig{
		Name:      "test-model",
		BlockSize: 16,
		AttentionGroups: []kvcache.AttentionGroupConfig{
			{WindowSize: 0, AttentionType: "full"},       // Group 0: Full Attention
			{WindowSize: 2048, AttentionType: "sliding"}, // Group 1: SWA-2048 (128 blocks)
		},
	}

	scorer := &kvcache.HybridPrefixCacheScorer{
		MediumWeights: mediumWeights,
		ModelConfig:   modelConfig,
	}

	// 128 blocks (2048 tokens)
	blockKeys := make([]kvblock.BlockHash, 128)
	hitmap := make(map[kvblock.BlockHash][]kvblock.PodEntry)

	for i := 0; i < 128; i++ {
		blockKeys[i] = kvblock.BlockHash(1000 + i)

		entries := []kvblock.PodEntry{}

		// pod-a: all 128 blocks on GPU, both groups
		entries = append(entries, kvblock.PodEntry{
			PodIdentifier: podA,
			DeviceTier:    "gpu",
			StoredGroups:  []int{0, 1},
		})

		// pod-b: all 128 blocks, mixed GPU/CPU, both groups
		tier := "gpu"
		if i >= 64 {
			tier = "cpu"
		}
		entries = append(entries, kvblock.PodEntry{
			PodIdentifier: podB,
			DeviceTier:    tier,
			StoredGroups:  []int{0, 1},
		})

		hitmap[blockKeys[i]] = entries
	}

	scored, err := scorer.Score(context.Background(), blockKeys, hitmap)
	assert.NoError(t, err)

	// pod-a: all GPU, both groups
	// Full: 128 × 2.0 = 256
	// SWA: 128 × 2.0 = 256
	// score = 0.7 × (256/128) + 0.3 × (256/128) = 0.7×2.0 + 0.3×2.0 = 2.0
	assert.Contains(t, scored, podA)
	assert.InDelta(t, 2.0, scored[podA], 0.01)

	// pod-b: mixed GPU/CPU, both groups
	// Full: (64 GPU × 2.0) + (64 CPU × 1.0) = 128 + 64 = 192
	// SWA: (64 GPU × 2.0) + (64 CPU × 1.0) = 192
	// score = 0.7 × (192/128) + 0.3 × (192/128) = 1.0 × 1.5 = 1.5
	assert.Contains(t, scored, podB)
	assert.InDelta(t, 1.5, scored[podB], 0.01)
}

func TestHybridPrefixCacheScorer_GroupFiltering(t *testing.T) {
	mediumWeights := map[string]float64{
		"gpu": 2.0,
	}

	modelConfig := &kvcache.ModelConfig{
		Name:      "test-model",
		BlockSize: 16,
		AttentionGroups: []kvcache.AttentionGroupConfig{
			{WindowSize: 0, AttentionType: "full"},       // Group 0: Full Attention
			{WindowSize: 1024, AttentionType: "sliding"}, // Group 1: SWA-1024 (64 blocks)
		},
	}

	scorer := &kvcache.HybridPrefixCacheScorer{
		MediumWeights: mediumWeights,
		ModelConfig:   modelConfig,
	}

	// 128 blocks
	blockKeys := make([]kvblock.BlockHash, 128)
	hitmap := make(map[kvblock.BlockHash][]kvblock.PodEntry)

	for i := 0; i < 128; i++ {
		blockKeys[i] = kvblock.BlockHash(1000 + i)

		entries := []kvblock.PodEntry{}

		// pod-a: has both groups for all blocks
		entries = append(entries, kvblock.PodEntry{
			PodIdentifier: podA,
			DeviceTier:    "gpu",
			StoredGroups:  []int{0, 1},
		})

		if i < 32 {
			// First 32 blocks: pod-b has both groups
			entries = append(entries, kvblock.PodEntry{
				PodIdentifier: podB,
				DeviceTier:    "gpu",
				StoredGroups:  []int{0, 1},
			})
		} else if i < 64 {
			// Blocks 32-63: pod-b only has group 1 (missing group 0!)
			entries = append(entries, kvblock.PodEntry{
				PodIdentifier: podB,
				DeviceTier:    "gpu",
				StoredGroups:  []int{1},
			})
		} else {
			// Blocks 64+: pod-b has both groups
			entries = append(entries, kvblock.PodEntry{
				PodIdentifier: podB,
				DeviceTier:    "gpu",
				StoredGroups:  []int{0, 1},
			})
		}

		hitmap[blockKeys[i]] = entries
	}

	scored, err := scorer.Score(context.Background(), blockKeys, hitmap)
	assert.NoError(t, err)

	// pod-a: has both groups for all blocks
	// Full (Group 0): 128 blocks × 2.0 = 256
	// SWA (Group 1, window = 64 blocks): 64 blocks × 2.0 = 128
	// score = 0.7 × (256/128) + 0.3 × (128/128) = 0.7×2.0 + 0.3×1.0 = 1.7
	assert.Contains(t, scored, podA)
	assert.InDelta(t, 1.7, scored[podA], 0.01)

	// pod-b: partial groups
	// Full (Group 0): Prefix stops at block 32 (block 32 missing Group 0)
	//   → 32 blocks × 2.0 = 64
	// SWA (Group 1, window = 64 blocks): Suffix from end
	//   - Blocks 127-64: all have Group 1 ✓ → 64 blocks × 2.0 = 128
	// score = 0.7 × (64/128) + 0.3 × (128/128) = 0.7×0.5 + 0.3×1.0 = 0.65
	assert.Contains(t, scored, podB)
	assert.InDelta(t, 0.65, scored[podB], 0.01)
}

func TestHybridPrefixCacheScorer_WindowConstraint(t *testing.T) {
	mediumWeights := map[string]float64{
		"gpu": 2.0,
	}

	modelConfig := &kvcache.ModelConfig{
		Name:      "test-model",
		BlockSize: 16,
		AttentionGroups: []kvcache.AttentionGroupConfig{
			{WindowSize: 0, AttentionType: "full"},       // Group 0: Full Attention
			{WindowSize: 1024, AttentionType: "sliding"}, // Group 1: SWA-1024 (64 blocks)
		},
	}

	scorer := &kvcache.HybridPrefixCacheScorer{
		MediumWeights: mediumWeights,
		ModelConfig:   modelConfig,
	}

	// Request 128 blocks, but sliding window limits SWA to 64
	blockKeys := make([]kvblock.BlockHash, 128)
	hitmap := make(map[kvblock.BlockHash][]kvblock.PodEntry)

	for i := 0; i < 128; i++ {
		blockKeys[i] = kvblock.BlockHash(1000 + i)
		hitmap[blockKeys[i]] = []kvblock.PodEntry{
			{PodIdentifier: podA, DeviceTier: "gpu", StoredGroups: []int{0, 1}},
		}
	}

	scored, err := scorer.Score(context.Background(), blockKeys, hitmap)
	assert.NoError(t, err)
	assert.Contains(t, scored, podA)

	// Full (Group 0): All 128 blocks × 2.0 = 256
	// SWA (Group 1, window = 64 blocks): 64 blocks × 2.0 = 128
	// score = 0.7 × (256/128) + 0.3 × (128/128) = 0.7×2.0 + 0.3×1.0 = 1.7
	assert.InDelta(t, 1.7, scored[podA], 0.01)
}

func TestHybridPrefixCacheScorer_FallbackToSimple(t *testing.T) {
	mediumWeights := map[string]float64{
		"gpu": 2.0,
	}

	// No attention groups - should fallback to simple longest prefix scoring
	modelConfig := &kvcache.ModelConfig{
		Name:            "test-model",
		BlockSize:       16,
		AttentionGroups: []kvcache.AttentionGroupConfig{},
	}

	scorer := &kvcache.HybridPrefixCacheScorer{
		MediumWeights: mediumWeights,
		ModelConfig:   modelConfig,
	}

	blockKeys := make([]kvblock.BlockHash, 64)
	hitmap := make(map[kvblock.BlockHash][]kvblock.PodEntry)

	for i := 0; i < 64; i++ {
		blockKeys[i] = kvblock.BlockHash(1000 + i)
		hitmap[blockKeys[i]] = []kvblock.PodEntry{
			{PodIdentifier: podA, DeviceTier: "gpu"},
		}
	}

	scored, err := scorer.Score(context.Background(), blockKeys, hitmap)
	assert.NoError(t, err)
	assert.Contains(t, scored, podA)

	// Fallback uses scoreLongestPrefix (cumulative weighted sum)
	// 64 blocks × 2.0 weight = 128.0
	assert.InDelta(t, 128.0, scored[podA], 0.01)
}

func TestHybridPrefixCacheScorer_SmallWindow(t *testing.T) {
	mediumWeights := map[string]float64{
		"gpu": 2.0,
	}

	modelConfig := &kvcache.ModelConfig{
		Name:      "test-model",
		BlockSize: 16,
		AttentionGroups: []kvcache.AttentionGroupConfig{
			{WindowSize: 0, AttentionType: "full"},      // Group 0: Full Attention
			{WindowSize: 512, AttentionType: "sliding"}, // Group 1: SWA-512 (32 blocks)
		},
	}

	scorer := &kvcache.HybridPrefixCacheScorer{
		MediumWeights: mediumWeights,
		ModelConfig:   modelConfig,
	}

	// Request 64 blocks (1024 tokens @ 16 tokens/block)
	blockKeys := make([]kvblock.BlockHash, 64)
	hitmap := make(map[kvblock.BlockHash][]kvblock.PodEntry)

	for i := 0; i < 64; i++ {
		blockKeys[i] = kvblock.BlockHash(1000 + i)
		hitmap[blockKeys[i]] = []kvblock.PodEntry{
			{PodIdentifier: podA, DeviceTier: "gpu", StoredGroups: []int{0, 1}},
		}
	}

	scored, err := scorer.Score(context.Background(), blockKeys, hitmap)
	assert.NoError(t, err)
	assert.Contains(t, scored, podA)

	// Full (Group 0): All 64 blocks × 2.0 = 128
	// SWA (Group 1, window = 512 tokens = 32 blocks): 32 blocks × 2.0 = 64
	// score = 0.7 × (128/64) + 0.3 × (64/64) = 0.7×2.0 + 0.3×1.0 = 1.7
	assert.InDelta(t, 1.7, scored[podA], 0.01)
}
