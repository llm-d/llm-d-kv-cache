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

package kvcache

import (
	"context"
	"fmt"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
)

// KVScoringStrategy defines the strategy used to score pods for KV cache block reuse.
type KVScoringStrategy string

const (
	// LongestPrefixMatch Score by longest consecutive match from start.
	LongestPrefixMatch KVScoringStrategy = "LongestPrefix"
)

// KVBlockScorerConfig holds the configuration for the KVBlockScorer.
type KVBlockScorerConfig struct {
	ScoringStrategy KVScoringStrategy
	BackendConfigs  []*KVCacheBackendConfig `json:"backendConfigs"`
}

// DefaultKVBlockScorerConfig returns the default configuration for the KVBlockScorer.
func DefaultKVBlockScorerConfig() *KVBlockScorerConfig {
	return &KVBlockScorerConfig{
		ScoringStrategy: LongestPrefixMatch,
		BackendConfigs:  DefaultKVCacheBackendConfig(),
	}
}

// KVBlockScorer defines the interface for implementing a KV block scoring
// strategy.
type KVBlockScorer interface {
	// Strategy returns the scoring strategy type.
	Strategy() KVScoringStrategy
	// Score scores the blocks based on the scoring strategy.
	// It returns a map of pod names to their scores.
	Score(ctx context.Context, keys []kvblock.BlockHash,
		keyToPods map[kvblock.BlockHash][]kvblock.PodEntry) (map[string]float64, error)
}

// NewKVBlockScorer creates a new KVBlockScorer based on the provided strategy.
func NewKVBlockScorer(config *KVBlockScorerConfig) (KVBlockScorer, error) {
	switch config.ScoringStrategy {
	case LongestPrefixMatch:
		// Build weight map from list of BackendConfigs for efficient lookup
		weightMap := make(map[string]float64)
		for _, medium := range config.BackendConfigs {
			weightMap[medium.Name] = medium.Weight
		}

		return &LongestPrefixScorer{
			MediumWeights: weightMap,
		}, nil
	default:
		return nil, fmt.Errorf("unsupported scoring strategy: %s", config.ScoringStrategy)
	}
}

// LongestPrefixScorer scores based on longest consecutive block matches count
// starting from block 0.
type LongestPrefixScorer struct {
	// mediumWeights maps medium/device tier names to their scoring weights
	MediumWeights map[string]float64
}

// Strategy returns the strategy type: LongestPrefixMatch.
func (s *LongestPrefixScorer) Strategy() KVScoringStrategy {
	return LongestPrefixMatch
}

// buildMaxWeights builds a map from podID to its maximum weight across all
// device tiers for the given entries. This replaces per-pod linear scans with
// a single pass over entries.
func buildMaxWeights(entries []kvblock.PodEntry, mediumWeights map[string]float64) map[string]float64 {
	maxWeights := make(map[string]float64, len(entries))
	for _, entry := range entries {
		weight := 1.0
		if mediumWeights != nil {
			if w, exists := mediumWeights[entry.DeviceTier]; exists {
				weight = w
			}
		}
		if cur, exists := maxWeights[entry.PodIdentifier]; !exists || weight > cur {
			maxWeights[entry.PodIdentifier] = weight
		}
	}
	return maxWeights
}

// Score implements the longest prefix scoring logic with weighted sum based on BackendConfig.
func (s *LongestPrefixScorer) Score(
	_ context.Context,
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
) (map[string]float64, error) {
	if len(keys) == 0 {
		return make(map[string]float64), nil
	}

	podScores := make(map[string]float64)

	// Build weight index for the first key in a single pass over entries.
	firstWeights := buildMaxWeights(keyToPods[keys[0]], s.MediumWeights)

	// activePods tracks pods still in the consecutive prefix chain.
	// Using a plain map and in-place deletion avoids allocating new sets
	// on every iteration.
	activePods := make(map[string]struct{}, len(firstWeights))
	for pod, w := range firstWeights {
		activePods[pod] = struct{}{}
		podScores[pod] = w
	}

	for i := 1; i < len(keys); i++ {
		if len(activePods) == 0 {
			break
		}

		// Build weight index for the current key once, then look up O(1) per pod.
		curWeights := buildMaxWeights(keyToPods[keys[i]], s.MediumWeights)

		// In-place intersection: delete pods from activePods that are not
		// in the current key, and accumulate scores for those that remain.
		for pod := range activePods {
			if w, exists := curWeights[pod]; exists {
				podScores[pod] += w
			} else {
				delete(activePods, pod)
			}
		}
	}

	// Return the map containing the final score for each pod encountered.
	return podScores, nil
}
