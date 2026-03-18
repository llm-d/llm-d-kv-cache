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

	scored, err := scorer.Score(context.Background(), blockKeys, hitmap, testModelName, 0)
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

	scored, err := scorer.Score(context.Background(), blockKeys, hitmap, testModelName, 0)
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

// TestHybridModelScorer_SmallRequest tests HMA scoring with a small request (512 tokens)
// that only needs Group 0 (full-attention).
func TestHybridModelScorer_SmallRequest(t *testing.T) {
	// Setup: DeepSeek-R1 style configuration
	// Group 0: Full-attention (always required)
	// Group 1: SWA-1024
	// Group 2: SWA-4096
	blockSize := 16
	window1024 := 1024
	window4096 := 4096

	config := &kvcache.KVBlockScorerConfig{
		ScoringStrategy: kvcache.HybridModel,
		BackendConfigs: []*kvcache.KVCacheBackendConfig{
			{
				Name:   "gpu",
				Weight: 1.0,
				ModelConfigs: map[string]*kvcache.ModelConfig{
					testModelName: {
						ModelName: testModelName,
						BlockSize: blockSize,
						AttentionGroups: map[int]*kvcache.AttentionGroupConfig{
							0: {WindowSize: nil},         // Full-attention
							1: {WindowSize: &window1024}, // SWA-1024
							2: {WindowSize: &window4096}, // SWA-4096
						},
					},
				},
			},
		},
	}

	scorer, err := kvcache.NewKVBlockScorer(config)
	assert.NoError(t, err)

	blockKeys := int64KeysToKVBlockKeys([]uint64{1001, 1002, 1003, 1004, 1005})

	// Pod-A: Evicted Groups 1, 2 (only Group 0 remains)
	// Pod-B: Nothing evicted (all groups available)
	// Pod-C: Evicted Group 0 (should be excluded)
	hitmap := map[kvblock.BlockHash][]kvblock.PodEntry{
		1001: {
			{PodIdentifier: "pod-a", DeviceTier: "gpu", EvictedGroups: []int{1, 2}},
			{PodIdentifier: "pod-b", DeviceTier: "gpu", EvictedGroups: nil}, // nil = nothing evicted
		},
		1002: {
			{PodIdentifier: "pod-a", DeviceTier: "gpu", EvictedGroups: []int{1, 2}},
			{PodIdentifier: "pod-b", DeviceTier: "gpu", EvictedGroups: nil},
		},
		1003: {
			{PodIdentifier: "pod-a", DeviceTier: "gpu", EvictedGroups: []int{1, 2}},
			{PodIdentifier: "pod-b", DeviceTier: "gpu", EvictedGroups: nil},
			{PodIdentifier: "pod-c", DeviceTier: "gpu", EvictedGroups: []int{0}}, // Evicted Group 0
		},
		1004: {
			{PodIdentifier: "pod-b", DeviceTier: "gpu", EvictedGroups: nil},
		},
		1005: {
			{PodIdentifier: "pod-b", DeviceTier: "gpu", EvictedGroups: nil},
		},
	}

	// Request: 512 tokens
	// Useful groups: [0] (only full-attention needed)
	// Pod-A: Base score = 3.0, has Group 0 → multiplier = 1.0 → final = 3.0
	// Pod-B: Base score = 5.0, has all groups → multiplier = 1.0 → final = 5.0
	// Pod-C: Base score = 1.0, missing Group 0 → multiplier = 0.0 → final = 0.0 (excluded)
	requestTokens := 512
	scores, err := scorer.Score(context.Background(), blockKeys, hitmap, testModelName, requestTokens)
	assert.NoError(t, err)

	expected := map[string]float64{
		"pod-a": 3.0, // Full hit for small request (only needs Group 0)
		"pod-b": 5.0, // Full hit
		"pod-c": 0.0, // Excluded (missing Group 0)
	}

	for pod, expectedScore := range expected {
		assert.InDelta(t, expectedScore, scores[pod], 0.0001, "Pod %s score mismatch", pod)
	}
}

// TestHybridModelScorer_MediumRequest tests HMA scoring with a medium request (2048 tokens)
// that needs Groups 0 and 1.
func TestHybridModelScorer_MediumRequest(t *testing.T) {
	blockSize := 16
	window1024 := 1024
	window4096 := 4096

	config := &kvcache.KVBlockScorerConfig{
		ScoringStrategy: kvcache.HybridModel,
		BackendConfigs: []*kvcache.KVCacheBackendConfig{
			{
				Name:   "gpu",
				Weight: 1.0,
				ModelConfigs: map[string]*kvcache.ModelConfig{
					testModelName: {
						ModelName: testModelName,
						BlockSize: blockSize,
						AttentionGroups: map[int]*kvcache.AttentionGroupConfig{
							0: {WindowSize: nil},
							1: {WindowSize: &window1024},
							2: {WindowSize: &window4096},
						},
					},
				},
			},
		},
	}

	scorer, err := kvcache.NewKVBlockScorer(config)
	assert.NoError(t, err)

	blockKeys := int64KeysToKVBlockKeys([]uint64{1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008})

	// Pod-A: Evicted Groups 1, 2 (only Group 0 remains)
	// Pod-B: Evicted Group 2 (Groups 0, 1 remain)
	// Pod-C: Nothing evicted (all groups available)
	hitmap := map[kvblock.BlockHash][]kvblock.PodEntry{
		1001: {
			{PodIdentifier: "pod-a", DeviceTier: "gpu", EvictedGroups: []int{1, 2}},
			{PodIdentifier: "pod-b", DeviceTier: "gpu", EvictedGroups: []int{2}},
			{PodIdentifier: "pod-c", DeviceTier: "gpu", EvictedGroups: nil},
		},
		1002: {
			{PodIdentifier: "pod-a", DeviceTier: "gpu", EvictedGroups: []int{1, 2}},
			{PodIdentifier: "pod-b", DeviceTier: "gpu", EvictedGroups: []int{2}},
			{PodIdentifier: "pod-c", DeviceTier: "gpu", EvictedGroups: nil},
		},
		1003: {
			{PodIdentifier: "pod-a", DeviceTier: "gpu", EvictedGroups: []int{1, 2}},
			{PodIdentifier: "pod-b", DeviceTier: "gpu", EvictedGroups: []int{2}},
			{PodIdentifier: "pod-c", DeviceTier: "gpu", EvictedGroups: nil},
		},
		1004: {
			{PodIdentifier: "pod-b", DeviceTier: "gpu", EvictedGroups: []int{2}},
			{PodIdentifier: "pod-c", DeviceTier: "gpu", EvictedGroups: nil},
		},
		1005: {
			{PodIdentifier: "pod-b", DeviceTier: "gpu", EvictedGroups: []int{2}},
			{PodIdentifier: "pod-c", DeviceTier: "gpu", EvictedGroups: nil},
		},
		1006: {
			{PodIdentifier: "pod-c", DeviceTier: "gpu", EvictedGroups: nil},
		},
		1007: {
			{PodIdentifier: "pod-c", DeviceTier: "gpu", EvictedGroups: nil},
		},
		1008: {
			{PodIdentifier: "pod-c", DeviceTier: "gpu", EvictedGroups: nil},
		},
	}

	// Request: 2048 tokens (> 1024, so needs Groups 0 and 1)
	// Useful groups: [0, 1]
	// Pod-A: Base score = 3.0, has 1/2 useful groups → multiplier = 0.3 + 0.7*(1/2) = 0.65 → final = 1.95
	// Pod-B: Base score = 5.0, has 2/2 useful groups → multiplier = 1.0 → final = 5.0
	// Pod-C: Base score = 8.0, has all groups → multiplier = 1.0 → final = 8.0
	requestTokens := 2048
	scores, err := scorer.Score(context.Background(), blockKeys, hitmap, testModelName, requestTokens)
	assert.NoError(t, err)

	expected := map[string]float64{
		"pod-a": 1.95, // Partial hit (1/2 useful groups)
		"pod-b": 5.0,  // Full hit
		"pod-c": 8.0,  // Full hit
	}

	for pod, expectedScore := range expected {
		assert.InDelta(t, expectedScore, scores[pod], 0.0001, "Pod %s score mismatch", pod)
	}
}

// TestHybridModelScorer_LargeRequest tests HMA scoring with a large request (6144 tokens)
// that needs all groups (0, 1, 2).
func TestHybridModelScorer_LargeRequest(t *testing.T) {
	blockSize := 16
	window1024 := 1024
	window4096 := 4096

	config := &kvcache.KVBlockScorerConfig{
		ScoringStrategy: kvcache.HybridModel,
		BackendConfigs: []*kvcache.KVCacheBackendConfig{
			{
				Name:   "gpu",
				Weight: 1.0,
				ModelConfigs: map[string]*kvcache.ModelConfig{
					testModelName: {
						ModelName: testModelName,
						BlockSize: blockSize,
						AttentionGroups: map[int]*kvcache.AttentionGroupConfig{
							0: {WindowSize: nil},
							1: {WindowSize: &window1024},
							2: {WindowSize: &window4096},
						},
					},
				},
			},
		},
	}

	scorer, err := kvcache.NewKVBlockScorer(config)
	assert.NoError(t, err)

	blockKeys := int64KeysToKVBlockKeys([]uint64{1001, 1002, 1003, 1004, 1005, 1006, 1007})

	// Pod-A: Evicted Groups 1, 2 (only Group 0 remains)
	// Pod-B: Evicted Group 2 (Groups 0, 1 remain)
	// Pod-C: Nothing evicted (all groups available)
	hitmap := map[kvblock.BlockHash][]kvblock.PodEntry{
		1001: {
			{PodIdentifier: "pod-a", DeviceTier: "gpu", EvictedGroups: []int{1, 2}},
			{PodIdentifier: "pod-b", DeviceTier: "gpu", EvictedGroups: []int{2}},
			{PodIdentifier: "pod-c", DeviceTier: "gpu", EvictedGroups: []int{}},
		},
		1002: {
			{PodIdentifier: "pod-a", DeviceTier: "gpu", EvictedGroups: []int{1, 2}},
			{PodIdentifier: "pod-b", DeviceTier: "gpu", EvictedGroups: []int{2}},
			{PodIdentifier: "pod-c", DeviceTier: "gpu", EvictedGroups: []int{}},
		},
		1003: {
			{PodIdentifier: "pod-a", DeviceTier: "gpu", EvictedGroups: []int{1, 2}},
			{PodIdentifier: "pod-b", DeviceTier: "gpu", EvictedGroups: []int{2}},
			{PodIdentifier: "pod-c", DeviceTier: "gpu", EvictedGroups: []int{}},
		},
		1004: {
			{PodIdentifier: "pod-b", DeviceTier: "gpu", EvictedGroups: []int{2}},
			{PodIdentifier: "pod-c", DeviceTier: "gpu", EvictedGroups: []int{}},
		},
		1005: {
			{PodIdentifier: "pod-b", DeviceTier: "gpu", EvictedGroups: []int{2}},
			{PodIdentifier: "pod-c", DeviceTier: "gpu", EvictedGroups: []int{}},
		},
		1006: {
			{PodIdentifier: "pod-c", DeviceTier: "gpu", EvictedGroups: []int{}},
		},
		1007: {
			{PodIdentifier: "pod-c", DeviceTier: "gpu", EvictedGroups: []int{}},
		},
	}

	// Request: 6144 tokens (> 4096, so needs all Groups 0, 1, 2)
	// Useful groups: [0, 1, 2]
	// Pod-A: Base score = 3.0, has 1/3 useful groups → multiplier = 0.3 + 0.7*(1/3) ≈ 0.533 → final ≈ 1.6
	// Pod-B: Base score = 5.0, has 2/3 useful groups → multiplier = 0.3 + 0.7*(2/3) ≈ 0.767 → final ≈ 3.83
	// Pod-C: Base score = 7.0, has 3/3 useful groups → multiplier = 1.0 → final = 7.0
	requestTokens := 6144
	scores, err := scorer.Score(context.Background(), blockKeys, hitmap, testModelName, requestTokens)
	assert.NoError(t, err)

	expected := map[string]float64{
		"pod-a": 3.0 * (0.3 + 0.7*(1.0/3.0)), // ≈ 1.6
		"pod-b": 5.0 * (0.3 + 0.7*(2.0/3.0)), // ≈ 3.833
		"pod-c": 7.0,                         // Full hit
	}

	for pod, expectedScore := range expected {
		assert.InDelta(t, expectedScore, scores[pod], 0.0001, "Pod %s score mismatch", pod)
	}
}

// TestHybridModelScorer_MissingGroupZero verifies that pods missing Group 0 are excluded.
func TestHybridModelScorer_MissingGroupZero(t *testing.T) {
	blockSize := 16
	window1024 := 1024

	config := &kvcache.KVBlockScorerConfig{
		ScoringStrategy: kvcache.HybridModel,
		BackendConfigs: []*kvcache.KVCacheBackendConfig{
			{
				Name:   "gpu",
				Weight: 1.0,
				ModelConfigs: map[string]*kvcache.ModelConfig{
					testModelName: {
						ModelName: testModelName,
						BlockSize: blockSize,
						AttentionGroups: map[int]*kvcache.AttentionGroupConfig{
							0: {WindowSize: nil},
							1: {WindowSize: &window1024},
						},
					},
				},
			},
		},
	}

	scorer, err := kvcache.NewKVBlockScorer(config)
	assert.NoError(t, err)

	blockKeys := int64KeysToKVBlockKeys([]uint64{1001, 1002, 1003})

	// Pod-A: Evicted Group 0 (only Group 1 remains - should be excluded)
	// Pod-B: Nothing evicted (Groups 0, 1 available)
	hitmap := map[kvblock.BlockHash][]kvblock.PodEntry{
		1001: {
			{PodIdentifier: "pod-a", DeviceTier: "gpu", EvictedGroups: []int{0}},
			{PodIdentifier: "pod-b", DeviceTier: "gpu", EvictedGroups: []int{}},
		},
		1002: {
			{PodIdentifier: "pod-a", DeviceTier: "gpu", EvictedGroups: []int{0}},
			{PodIdentifier: "pod-b", DeviceTier: "gpu", EvictedGroups: []int{}},
		},
		1003: {
			{PodIdentifier: "pod-a", DeviceTier: "gpu", EvictedGroups: []int{0}},
			{PodIdentifier: "pod-b", DeviceTier: "gpu", EvictedGroups: []int{}},
		},
	}

	requestTokens := 2048
	scores, err := scorer.Score(context.Background(), blockKeys, hitmap, testModelName, requestTokens)
	assert.NoError(t, err)

	// Pod-A should be excluded (multiplier = 0.0)
	assert.Equal(t, 0.0, scores["pod-a"], "Pod-A should be excluded (missing Group 0)")
	assert.Greater(t, scores["pod-b"], 0.0, "Pod-B should have positive score")
}

// TestHybridModelScorer_UnknownModel verifies graceful fallback for models not in registry.
func TestHybridModelScorer_UnknownModel(t *testing.T) {
	blockSize := 16
	window1024 := 1024

	config := &kvcache.KVBlockScorerConfig{
		ScoringStrategy: kvcache.HybridModel,
		BackendConfigs: []*kvcache.KVCacheBackendConfig{
			{
				Name:   "gpu",
				Weight: 1.0,
				ModelConfigs: map[string]*kvcache.ModelConfig{
					"deepseek-r1": {
						ModelName: "deepseek-r1",
						BlockSize: blockSize,
						AttentionGroups: map[int]*kvcache.AttentionGroupConfig{
							0: {WindowSize: nil},
							1: {WindowSize: &window1024},
						},
					},
				},
			},
		},
	}

	scorer, err := kvcache.NewKVBlockScorer(config)
	assert.NoError(t, err)

	blockKeys := int64KeysToKVBlockKeys([]uint64{1001, 1002, 1003})

	hitmap := map[kvblock.BlockHash][]kvblock.PodEntry{
		1001: {{PodIdentifier: podA, DeviceTier: "gpu"}},
		1002: {{PodIdentifier: podA, DeviceTier: "gpu"}},
		1003: {{PodIdentifier: podA, DeviceTier: "gpu"}},
	}

	// Score with unknown model - should fallback to LongestPrefix scoring
	scores, err := scorer.Score(context.Background(), blockKeys, hitmap, "unknown-model", 2048)
	assert.NoError(t, err)
	assert.Equal(t, 3.0, scores[podA], "Should use LongestPrefix scoring for unknown model")
}

// TestHybridModelScorer_StandardModel verifies fallback for models without attention groups.
func TestHybridModelScorer_StandardModel(t *testing.T) {
	config := &kvcache.KVBlockScorerConfig{
		ScoringStrategy: kvcache.HybridModel,
		BackendConfigs: []*kvcache.KVCacheBackendConfig{
			{
				Name:   "gpu",
				Weight: 1.0,
				ModelConfigs: map[string]*kvcache.ModelConfig{
					"llama-3": {
						ModelName:       "llama-3",
						BlockSize:       16,
						AttentionGroups: nil, // No attention groups = standard model
					},
				},
			},
		},
	}

	scorer, err := kvcache.NewKVBlockScorer(config)
	assert.NoError(t, err)

	blockKeys := int64KeysToKVBlockKeys([]uint64{1001, 1002, 1003})

	hitmap := map[kvblock.BlockHash][]kvblock.PodEntry{
		1001: {{PodIdentifier: podA, DeviceTier: "gpu", EvictedGroups: []int{1, 2}}},
		1002: {{PodIdentifier: podA, DeviceTier: "gpu", EvictedGroups: []int{1, 2}}},
		1003: {{PodIdentifier: podA, DeviceTier: "gpu", EvictedGroups: []int{1, 2}}},
	}

	// Score standard model - should ignore EvictedGroups and use LongestPrefix
	scores, err := scorer.Score(context.Background(), blockKeys, hitmap, "llama-3", 2048)
	assert.NoError(t, err)
	assert.Equal(t, 3.0, scores[podA], "Should use LongestPrefix scoring for standard model")
}

// TestHybridModelScorer_NoModelConfigs verifies backward compatibility when HybridModel
// strategy is configured but no ModelConfigs are provided (should fallback to LongestPrefix).
func TestHybridModelScorer_NoModelConfigs(t *testing.T) {
	config := &kvcache.KVBlockScorerConfig{
		ScoringStrategy: kvcache.HybridModel,
		BackendConfigs: []*kvcache.KVCacheBackendConfig{
			{
				Name:   "gpu",
				Weight: 1.0,
				// No ModelConfigs provided
			},
		},
	}

	scorer, err := kvcache.NewKVBlockScorer(config)
	assert.NoError(t, err, "Should create scorer even without ModelConfigs")

	blockKeys := int64KeysToKVBlockKeys([]uint64{1001, 1002, 1003})

	hitmap := map[kvblock.BlockHash][]kvblock.PodEntry{
		1001: {{PodIdentifier: podA, DeviceTier: "gpu", EvictedGroups: []int{1, 2}}},
		1002: {{PodIdentifier: podA, DeviceTier: "gpu", EvictedGroups: []int{1, 2}}},
		1003: {{PodIdentifier: podA, DeviceTier: "gpu", EvictedGroups: []int{1, 2}}},
	}

	// Should fallback to LongestPrefix scoring (ignore EvictedGroups)
	scores, err := scorer.Score(context.Background(), blockKeys, hitmap, "any-model", 2048)
	assert.NoError(t, err)
	assert.Equal(t, 3.0, scores[podA], "Should use LongestPrefix scoring when no ModelConfigs provided")
}

// TestHybridModelScorer_MixedModels verifies handling multiple model types in same scorer.
func TestHybridModelScorer_MixedModels(t *testing.T) {
	blockSize := 16
	window1024 := 1024
	window4096 := 4096

	config := &kvcache.KVBlockScorerConfig{
		ScoringStrategy: kvcache.HybridModel,
		BackendConfigs: []*kvcache.KVCacheBackendConfig{
			{
				Name:   "gpu",
				Weight: 1.0,
				ModelConfigs: map[string]*kvcache.ModelConfig{
					"deepseek-r1": {
						ModelName: "deepseek-r1",
						BlockSize: blockSize,
						AttentionGroups: map[int]*kvcache.AttentionGroupConfig{
							0: {WindowSize: nil},
							1: {WindowSize: &window1024},
							2: {WindowSize: &window4096},
						},
					},
					"llama-3": {
						ModelName:       "llama-3",
						BlockSize:       16,
						AttentionGroups: nil, // Standard model
					},
				},
			},
		},
	}

	scorer, err := kvcache.NewKVBlockScorer(config)
	assert.NoError(t, err)

	blockKeys := int64KeysToKVBlockKeys([]uint64{1001, 1002, 1003})

	hitmap := map[kvblock.BlockHash][]kvblock.PodEntry{
		1001: {{PodIdentifier: podA, DeviceTier: "gpu", EvictedGroups: []int{1, 2}}},
		1002: {{PodIdentifier: podA, DeviceTier: "gpu", EvictedGroups: []int{1, 2}}},
		1003: {{PodIdentifier: podA, DeviceTier: "gpu", EvictedGroups: []int{1, 2}}},
	}

	// Test 1: HMA model (deepseek-r1) with large request - should apply group penalty
	requestTokens := 6144 // Needs all groups
	scores, err := scorer.Score(context.Background(), blockKeys, hitmap, "deepseek-r1", requestTokens)
	assert.NoError(t, err)
	expectedScore := 3.0 * (0.3 + 0.7*(1.0/3.0)) // Only has 1/3 useful groups
	assert.InDelta(t, expectedScore, scores[podA], 0.0001, "Should apply HMA penalty for deepseek-r1")

	// Test 2: Standard model (llama-3) - should ignore EvictedGroups
	scores, err = scorer.Score(context.Background(), blockKeys, hitmap, "llama-3", requestTokens)
	assert.NoError(t, err)
	assert.Equal(t, 3.0, scores[podA], "Should use standard scoring for llama-3")
}
