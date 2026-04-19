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
	// HybridPrefixMatch Score for HMA models with separate full attention and SWA scoring.
	HybridPrefixMatch KVScoringStrategy = "HybridPrefix"
)

// KVBlockScorerConfig holds the configuration for the KVBlockScorer.
type KVBlockScorerConfig struct {
	ScoringStrategy KVScoringStrategy
	BackendConfigs  []*KVCacheBackendConfig `json:"backendConfigs"`
	ModelRegistry   *ModelRegistry          `json:"-"`
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
	// modelName is used by HMA scorers to determine attention group configuration.
	// It returns a map of pod names to their scores.
	Score(ctx context.Context, keys []kvblock.BlockHash,
		keyToPods map[kvblock.BlockHash][]kvblock.PodEntry, modelName string) (map[string]float64, error)
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
	case HybridPrefixMatch:
		// Build weight map from list of BackendConfigs for efficient lookup
		weightMap := make(map[string]float64)
		for _, medium := range config.BackendConfigs {
			weightMap[medium.Name] = medium.Weight
		}

		return &HybridPrefixCacheScorer{
			MediumWeights: weightMap,
			ModelRegistry: config.ModelRegistry,
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
	_ string,
) (map[string]float64, error) {
	if len(keys) == 0 {
		return make(map[string]float64), nil
	}

	podScores := make(map[string]float64)

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
		podScores[pod] = w
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
				podScores[pod] += w
			} else {
				delete(activePods, pod)
			}
		}
	}

	// Return the map containing the final score for each pod encountered.
	return podScores, nil
}

// HybridPrefixCacheScorer scores HMA models with multiple attention groups using a
// single-pass cache boundary evaluator per pod.
// Group 0 (full attention) acts as a kill switch — iteration terminates on a miss.
// SWA groups track contiguous block counts with sticky last_seq checkpoints.
// Final score = min(all group checkpoints) + 1 (effective cached block count).
type HybridPrefixCacheScorer struct {
	MediumWeights map[string]float64
	ModelRegistry *ModelRegistry
}

// Strategy returns the strategy type: HybridPrefixMatch.
func (s *HybridPrefixCacheScorer) Strategy() KVScoringStrategy {
	return HybridPrefixMatch
}

// Score evaluates each candidate pod using a single-pass boundary check.
func (s *HybridPrefixCacheScorer) Score(
	ctx context.Context,
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
	modelName string,
) (map[string]float64, error) {
	if len(keys) == 0 {
		return make(map[string]float64), nil
	}

	info := s.ModelRegistry.GetAttentionInfo(modelName)
	if info == nil {
		fallback := &LongestPrefixScorer{MediumWeights: s.MediumWeights}
		return fallback.Score(ctx, keys, keyToPods, modelName)
	}

	podScores := make(map[string]float64)
	for _, entry := range keyToPods[keys[0]] {
		if containsGroup(entry.StoredGroups, info.FullGroupID) {
			score := evaluatePod(keys, keyToPods, entry.PodIdentifier,
				info.FullGroupID, info.SWAGroupIDs, info.SWAWindowBlocks)
			if score > 0 {
				podScores[entry.PodIdentifier] = score
			}
		}
	}

	return podScores, nil
}

// evaluatePod runs a single-pass boundary evaluation for one pod.
// Walks left-to-right through requested blocks. Full attention (group 0) is the
// kill switch — loop terminates on a miss. SWA groups track contiguous counts
// with sticky last_seq. Returns min(all checkpoints) + 1, or 0 if no valid cache.
func evaluatePod(
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
	pod string,
	fullGroupID int,
	swaGroupIDs []int,
	swaWindowBlocks []int,
) float64 {
	numSWA := len(swaGroupIDs)

	lastSeqFull := -1
	swaCounts := make([]int, numSWA)
	swaLastSeqs := make([]int, numSWA)
	for i := range swaLastSeqs {
		swaLastSeqs[i] = -1
	}

	for b := 0; b < len(keys); b++ {
		entries := keyToPods[keys[b]]

		if !podHasBlockInGroup(entries, pod, fullGroupID) {
			break
		}
		lastSeqFull = b

		for g := 0; g < numSWA; g++ {
			if podHasBlockInGroup(entries, pod, swaGroupIDs[g]) {
				swaCounts[g]++
				if swaCounts[g] >= swaWindowBlocks[g] {
					swaLastSeqs[g] = b
				}
			} else {
				swaCounts[g] = 0
			}
		}
	}

	if lastSeqFull < 0 {
		return 0
	}

	checkpoint := lastSeqFull
	for g := 0; g < numSWA; g++ {
		if swaLastSeqs[g] < 0 {
			return 0
		}
		if swaLastSeqs[g] < checkpoint {
			checkpoint = swaLastSeqs[g]
		}
	}

	return float64(checkpoint + 1)
}

// podHasBlockInGroup checks if a specific pod has a block entry containing the given group ID.
func podHasBlockInGroup(entries []kvblock.PodEntry, podID string, groupID int) bool {
	for _, entry := range entries {
		if entry.PodIdentifier == podID && containsGroup(entry.StoredGroups, groupID) {
			return true
		}
	}
	return false
}

// containsGroup checks if a group ID exists in the StoredGroups bitmask.
func containsGroup(storedGroups uint32, groupID int) bool {
	return storedGroups&(1<<groupID) != 0
}
