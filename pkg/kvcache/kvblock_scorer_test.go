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

// TestLongestPrefixScorer_UnknownTierDefaultsToZero verifies that unknown device
// tiers default to weight 0 instead of 1.0, preventing silent score inflation.
func TestLongestPrefixScorer_UnknownTierDefaultsToZero(t *testing.T) {
	mediumWeights := map[string]float64{
		"gpu": 1.0,
		"cpu": 0.8,
	}

	scorer := &kvcache.LongestPrefixScorer{
		MediumWeights: mediumWeights,
	}
	blockKeys := int64KeysToKVBlockKeys([]uint64{1001, 1002, 1003})

	hitmap := map[kvblock.BlockHash][]kvblock.PodEntry{
		1001: {
			{PodIdentifier: podA, DeviceTier: "gpu"},
			{PodIdentifier: podB, DeviceTier: "fs"}, // unknown tier
		},
		1002: {
			{PodIdentifier: podA, DeviceTier: "cpu"},
			{PodIdentifier: podB, DeviceTier: "fs"}, // unknown tier
		},
		1003: {
			{PodIdentifier: podB, DeviceTier: "fs"}, // unknown tier
		},
	}

	// podA: gpu(1.0) + cpu(0.8) = consecutive prefix match for 2 blocks -> 1.8
	// podB: fs is unknown -> weight 0 for all blocks -> score 0
	expected := map[string]float64{
		podA: 1.8,
		podB: 0.0,
	}

	scored, err := scorer.Score(context.Background(), blockKeys, hitmap)
	assert.NoError(t, err)
	assert.Len(t, scored, 2)
	for pod, score := range scored {
		assert.InDelta(t, expected[pod], score, 0.0001,
			"pod %s: expected %f, got %f", pod, expected[pod], score)
	}
}

// TestLongestPrefixScorer_UnknownTierDoesNotInflateScore verifies that an unknown
// tier doesn't artificially inflate a pod's score above known-tier pods.
func TestLongestPrefixScorer_UnknownTierDoesNotInflateScore(t *testing.T) {
	mediumWeights := map[string]float64{
		"gpu": 1.0,
		"cpu": 0.8,
	}

	scorer := &kvcache.LongestPrefixScorer{
		MediumWeights: mediumWeights,
	}
	blockKeys := int64KeysToKVBlockKeys([]uint64{1001, 1002})

	hitmap := map[kvblock.BlockHash][]kvblock.PodEntry{
		1001: {
			{PodIdentifier: "gpu-pod", DeviceTier: "gpu"},
			{PodIdentifier: "fs-pod", DeviceTier: "fs"}, // unknown tier
		},
		1002: {
			{PodIdentifier: "gpu-pod", DeviceTier: "gpu"},
			{PodIdentifier: "fs-pod", DeviceTier: "fs"}, // unknown tier
		},
	}

	scored, err := scorer.Score(context.Background(), blockKeys, hitmap)
	assert.NoError(t, err)

	// gpu-pod: 2 blocks × 1.0 = 2.0
	// fs-pod: 2 blocks × 0 (unknown) = 0
	expectedGPU := 2.0
	expectedFS := 0.0

	assert.InDelta(t, expectedGPU, scored["gpu-pod"], 0.0001,
		"gpu-pod should score 2.0 (2 blocks × 1.0)")
	assert.InDelta(t, expectedFS, scored["fs-pod"], 0.0001,
		"fs-pod should score 0 (unknown tier defaults to 0)")
	assert.Greater(t, scored["gpu-pod"], scored["fs-pod"],
		"gpu-pod (known tier) should score higher than fs-pod (unknown tier)")
}
