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
	"k8s.io/apimachinery/pkg/util/sets"
)

// KVScoringStrategy defines the strategy used to score pods for KV cache block reuse.
type KVScoringStrategy string

const (
	// LongestPrefixMatch Score by longest consecutive match from start.
	LongestPrefixMatch KVScoringStrategy = "LongestPrefix"
	// HybridModel Score with HMA-aware group validation for hybrid attention models.
	HybridModel KVScoringStrategy = "HybridModel"
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
	// modelName identifies which model configuration to use (for HMA scoring).
	// requestTokens is the total number of tokens in the request (used for HMA scoring).
	Score(ctx context.Context, keys []kvblock.BlockHash,
		keyToPods map[kvblock.BlockHash][]kvblock.PodEntry, modelName string, requestTokens int) (map[string]float64, error)
}

// NewKVBlockScorer creates a new KVBlockScorer based on the provided strategy.
func NewKVBlockScorer(config *KVBlockScorerConfig) (KVBlockScorer, error) {
	// Build weight map from list of BackendConfigs for efficient lookup
	weightMap := make(map[string]float64)
	for _, medium := range config.BackendConfigs {
		weightMap[medium.Name] = medium.Weight
	}

	switch config.ScoringStrategy {
	case LongestPrefixMatch:
		return &LongestPrefixScorer{
			MediumWeights: weightMap,
		}, nil
	case HybridModel:
		// Build model registry from backend configs
		modelConfigs := make(map[string]*ModelConfig)
		for _, backend := range config.BackendConfigs {
			if backend.ModelConfigs != nil {
				for modelName, modelConfig := range backend.ModelConfigs {
					// Use first occurrence of model config (backends should have consistent configs)
					if _, exists := modelConfigs[modelName]; !exists {
						modelConfigs[modelName] = modelConfig
					}
				}
			}
		}

		// If no model configs provided, fallback to LongestPrefix scorer
		// This maintains backward compatibility with configs that don't have model registry
		if len(modelConfigs) == 0 {
			return &LongestPrefixScorer{
				MediumWeights: weightMap,
			}, nil
		}

		return &HybridModelScorer{
			baseScorer: &LongestPrefixScorer{
				MediumWeights: weightMap,
			},
			modelConfigs: modelConfigs,
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
	_ string, // modelName - not used by LongestPrefixScorer
	_ int, // requestTokens - not used by LongestPrefixScorer
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

// HybridModelScorer wraps LongestPrefixScorer with HMA-aware group validation.
// It computes base scores using the longest prefix logic, then applies a multiplier
// based on which attention groups are cached vs. required for the request.
type HybridModelScorer struct {
	baseScorer   *LongestPrefixScorer
	modelConfigs map[string]*ModelConfig // model name -> model config
}

// Strategy returns HybridModel.
func (h *HybridModelScorer) Strategy() KVScoringStrategy {
	return HybridModel
}

// Score applies HMA-aware scoring with group coverage multipliers.
// For models not in the registry or without attention groups, falls back to LongestPrefix scoring.
func (h *HybridModelScorer) Score(
	ctx context.Context,
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
	modelName string,
	requestTokens int,
) (map[string]float64, error) {
	// Step 0: Look up model configuration
	modelConfig, exists := h.modelConfigs[modelName]

	// Fallback to standard scoring if:
	// 1. Model not found in registry (unknown model)
	// 2. Model has no attention groups configured (standard model)
	if !exists || modelConfig.AttentionGroups == nil || len(modelConfig.AttentionGroups) == 0 {
		return h.baseScorer.Score(ctx, keys, keyToPods, modelName, requestTokens)
	}

	// Step 1: Get base scores from LongestPrefixScorer
	baseScores, err := h.baseScorer.Score(ctx, keys, keyToPods, modelName, requestTokens)
	if err != nil {
		return nil, err
	}

	// Step 2: Determine useful groups for this request
	usefulGroups := h.computeUsefulGroups(requestTokens, modelConfig)

	// Step 3: Build pod -> cached groups mapping
	// We need to examine the PodEntry.CachedGroups field for each pod
	podCachedGroups := h.buildPodCachedGroupsMap(keyToPods, modelConfig)

	// Step 4: Apply group coverage multipliers
	finalScores := make(map[string]float64)
	for podID, baseScore := range baseScores {
		multiplier := h.computeGroupCoverageMultiplier(podID, usefulGroups, podCachedGroups)
		finalScores[podID] = baseScore * multiplier
	}

	return finalScores, nil
}

// computeUsefulGroups returns the set of attention groups needed for this request.
// Group 0 (full-attention) is always required.
// SWA groups are useful if requestTokens > windowSize.
func (h *HybridModelScorer) computeUsefulGroups(requestTokens int, modelConfig *ModelConfig) sets.Set[int] {
	useful := sets.New[int]()

	for groupID, config := range modelConfig.AttentionGroups {
		if config.WindowSize == nil {
			// Full-attention group - always useful
			useful.Insert(groupID)
		} else if requestTokens > *config.WindowSize {
			// SWA group is useful if request exceeds window
			useful.Insert(groupID)
		}
	}

	return useful
}

// buildPodCachedGroupsMap examines all PodEntries to determine which groups
// each pod has available (cached). Computed as: available = allGroups - evictedGroups.
func (h *HybridModelScorer) buildPodCachedGroupsMap(
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
	modelConfig *ModelConfig,
) map[string]sets.Set[int] {
	podGroups := make(map[string]sets.Set[int])

	// Get all groups for this model
	allGroups := sets.New[int]()
	for groupID := range modelConfig.AttentionGroups {
		allGroups.Insert(groupID)
	}

	for _, entries := range keyToPods {
		for _, entry := range entries {
			if _, exists := podGroups[entry.PodIdentifier]; !exists {
				// Compute available groups = allGroups - evictedGroups
				available := allGroups.Clone()

				if len(entry.EvictedGroups) > 0 {
					// Remove evicted groups from available
					for _, evictedGroup := range entry.EvictedGroups {
						available.Delete(evictedGroup)
					}
				}
				// else: nil or empty EvictedGroups = nothing evicted = all available

				podGroups[entry.PodIdentifier] = available
			}
		}
	}

	return podGroups
}

// computeGroupCoverageMultiplier calculates the score multiplier based on
// which useful groups are cached by the pod.
//
// Rules:
// - If Group 0 (full-attention) is missing: multiplier = 0.0 (exclude pod)
// - If all useful groups are cached: multiplier = 1.0 (full hit)
// - If some useful groups are cached: multiplier = 0.3 + 0.7 * (cached/total)
//
// This rewards partial cache coverage while ensuring full-attention is required.
func (h *HybridModelScorer) computeGroupCoverageMultiplier(
	podID string,
	usefulGroups sets.Set[int],
	podCachedGroups map[string]sets.Set[int],
) float64 {
	cachedGroups, hasCachedInfo := podCachedGroups[podID]
	if !hasCachedInfo {
		// No cached info for this pod - treat as full miss
		return 0.0
	}

	// Group 0 (full-attention) is ALWAYS required
	if !cachedGroups.Has(0) {
		return 0.0 // Exclude pod entirely
	}

	// Calculate how many useful groups are cached
	cachedUsefulCount := 0
	for groupID := range usefulGroups {
		if cachedGroups.Has(groupID) {
			cachedUsefulCount++
		}
	}

	totalUsefulCount := usefulGroups.Len()
	if totalUsefulCount == 0 {
		// No useful groups needed (shouldn't happen, but handle gracefully)
		return 1.0
	}

	if cachedUsefulCount == totalUsefulCount {
		// Full hit - all useful groups cached
		return 1.0
	}

	// Partial hit - linear scaling from 0.3 to 1.0
	// Minimum 0.3 ensures pods with Group 0 are preferred over cold starts
	ratio := float64(cachedUsefulCount) / float64(totalUsefulCount)
	return 0.3 + (0.7 * ratio)
}
