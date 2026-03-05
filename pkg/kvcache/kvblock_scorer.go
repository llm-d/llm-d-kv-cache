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

// ScorerResult contains the scoring results including both weighted scores
// and unweighted consecutive match counts.
type ScorerResult struct {
	// WeightedScores maps pod identifiers to their weighted scores.
	WeightedScores map[string]float64
	// ConsecutiveMatches maps pod identifiers to the number of consecutive
	// blocks matched from the start (unweighted count).
	ConsecutiveMatches map[string]int
}

// KVBlockScorer defines the interface for implementing a KV block scoring
// strategy.
type KVBlockScorer interface {
	// Strategy returns the scoring strategy type.
	Strategy() KVScoringStrategy
	// Score scores the blocks based on the scoring strategy.
	// It returns both weighted scores and unweighted consecutive match counts.
	Score(ctx context.Context, keys []kvblock.BlockHash,
		keyToPods map[kvblock.BlockHash][]kvblock.PodEntry) (*ScorerResult, error)
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

// fillMaxWeights populates dst with the maximum weight per podID across all
// device tiers for the given entries. The caller must clear dst before calling.
func fillMaxWeights(dst map[string]float64, entries []kvblock.PodEntry, mediumWeights map[string]float64) {
	for _, entry := range entries {
		weight := 1.0
		if mediumWeights != nil {
			if w, exists := mediumWeights[entry.DeviceTier]; exists {
				weight = w
			}
		}
		if cur, exists := dst[entry.PodIdentifier]; !exists || weight > cur {
			dst[entry.PodIdentifier] = weight
		}
	}
}

// Score implements the longest prefix scoring logic with weighted sum based on BackendConfig.
func (s *LongestPrefixScorer) Score(
	_ context.Context,
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
) (*ScorerResult, error) {
	weightedScores := make(map[string]float64)
	consecutiveMatches := make(map[string]int)

	if len(keys) == 0 {
		return &ScorerResult{
			WeightedScores:     make(map[string]float64),
			ConsecutiveMatches: make(map[string]int),
		}, nil
	}

	// Scratch map reused across iterations to avoid per-key allocation.
	curWeights := make(map[string]float64)

	// Build weight index for the first key in a single pass over entries.
	fillMaxWeights(curWeights, keyToPods[keys[0]], s.MediumWeights)

	// activePods tracks pods still in the consecutive prefix chain.
	// Using a plain map and in-place deletion avoids allocating new sets
	// on every iteration.
	activePods := make(map[string]struct{}, len(curWeights))
	for pod, w := range curWeights {
		activePods[pod] = struct{}{}
		weightedScores[pod] = w
		consecutiveMatches[pod] = 1
	}

	for i := 1; i < len(keys); i++ {
		if len(activePods) == 0 {
			break
		}

		// Reuse scratch map: clear and refill for current key.
		clear(curWeights)
		fillMaxWeights(curWeights, keyToPods[keys[i]], s.MediumWeights)

		// In-place intersection: delete pods from activePods that are not
		// in the current key, and accumulate scores for those that remain.
		for pod := range activePods {
			if w, exists := curWeights[pod]; exists {
				weightedScores[pod] += w
				consecutiveMatches[pod]++
			} else {
				delete(activePods, pod)
			}
		}
	}

	// Return the result containing both weighted scores and consecutive match counts.
	return &ScorerResult{
		WeightedScores:     weightedScores,
		ConsecutiveMatches: consecutiveMatches,
	}, nil
}
