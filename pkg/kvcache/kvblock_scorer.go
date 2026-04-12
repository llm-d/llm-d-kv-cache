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
	// HybridPrefixMatch Score by longest prefix considering multiple attention groups.
	HybridPrefixMatch KVScoringStrategy = "HybridPrefix"
)

// KVBlockScorerConfig holds the configuration for the KVBlockScorer.
type KVBlockScorerConfig struct {
	ScoringStrategy KVScoringStrategy
	BackendConfigs  []*KVCacheBackendConfig `json:"backendConfigs"`
	ModelConfig     *ModelConfig            // Model config for hybrid attention scoring
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
	case HybridPrefixMatch:
		if config.ModelConfig == nil {
			return nil, fmt.Errorf("HybridPrefixMatch requires ModelConfig")
		}
		return &HybridPrefixCacheScorer{
			MediumWeights: weightMap,
			ModelConfig:   config.ModelConfig,
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

// HybridPrefixCacheScorer implements hybrid attention scoring using a fixed-point
// algorithm to find the longest cache hit across multiple attention groups.
// Based on vLLM's find_longest_cache_hit algorithm.
type HybridPrefixCacheScorer struct {
	// MediumWeights maps medium/device tier names to their scoring weights
	MediumWeights map[string]float64
	// ModelConfig holds the model-specific configuration with attention groups
	ModelConfig *ModelConfig
}

// Strategy returns the strategy type: HybridPrefixMatch.
func (s *HybridPrefixCacheScorer) Strategy() KVScoringStrategy {
	return HybridPrefixMatch
}

// Score implements hybrid prefix scoring with weighted combination of Full and SWA scores.
// Algorithm:
// 1. Identify Full Attention groups (primary) and SWA groups (secondary)
// 2. Full Attention: Score longest consecutive prefix from start (left-to-right)
// 3. SWA: Score longest consecutive suffix from end (right-to-left, up to window size)
// 4. Normalize: score = 0.7 × fullHitRatio + 0.3 × swaHitRatio
//
// This matches vLLM's scoring logic where:
// - Full Attention requires consecutive prefix (strict)
// - SWA only cares about recent context (tail matching)
func (s *HybridPrefixCacheScorer) Score(
	ctx context.Context,
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
) (map[string]float64, error) {
	if len(keys) == 0 {
		return make(map[string]float64), nil
	}

	if s.ModelConfig == nil || len(s.ModelConfig.AttentionGroups) == 0 {
		// Fallback to simple longest prefix if no attention groups configured
		return s.scoreLongestPrefix(ctx, keys, keyToPods)
	}

	// Identify Full Attention (primary) and SWA (secondary) groups
	fullGroups := make([]int, 0)
	swaGroups := make([]int, 0)

	for idx, group := range s.ModelConfig.AttentionGroups {
		if group.AttentionType == "full" || group.WindowSize == 0 {
			fullGroups = append(fullGroups, idx)
		} else {
			swaGroups = append(swaGroups, idx)
		}
	}

	// If no groups identified, fallback to simple prefix scoring
	if len(fullGroups) == 0 && len(swaGroups) == 0 {
		return s.scoreLongestPrefix(ctx, keys, keyToPods)
	}

	return s.scoreWithWeightedCombination(keys, keyToPods, fullGroups, swaGroups)
}

// scoreWithWeightedCombination implements weighted combination scoring:
// score = 0.7 × fullHitRatio + 0.3 × swaHitRatio
//
// Full Attention: Longest consecutive prefix from start (left-to-right)
// SWA: Longest consecutive suffix from end (right-to-left, up to window size)
func (s *HybridPrefixCacheScorer) scoreWithWeightedCombination(
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
	fullGroups []int,
	swaGroups []int,
) (map[string]float64, error) {
	totalBlocks := len(keys)
	maxPossibleScore := float64(totalBlocks)

	// Collect all pods
	allPods := make(map[string]struct{})
	for _, entries := range keyToPods {
		for _, entry := range entries {
			allPods[entry.PodIdentifier] = struct{}{}
		}
	}

	podScores := make(map[string]float64)

	for podID := range allPods {
		var fullScore float64
		var swaScore float64

		// Score Full Attention groups (prefix matching from start)
		if len(fullGroups) > 0 {
			fullScore = s.scoreFullAttentionForPod(keys, keyToPods, podID, fullGroups)
		}

		// Score SWA groups (suffix matching from end)
		if len(swaGroups) > 0 {
			swaScore = s.scoreSWAForPod(keys, keyToPods, podID, swaGroups)
		}

		// Weighted combination: 0.7 × full + 0.3 × swa
		// Normalize by max possible score
		fullRatio := fullScore / maxPossibleScore
		swaRatio := swaScore / maxPossibleScore

		normalizedScore := 0.7*fullRatio + 0.3*swaRatio

		if normalizedScore > 0 {
			podScores[podID] = normalizedScore
		}
	}

	return podScores, nil
}

// scoreFullAttentionForPod scores Full Attention groups using longest prefix matching.
// Returns weighted sum of matching blocks from start.
func (s *HybridPrefixCacheScorer) scoreFullAttentionForPod(
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
	podID string,
	fullGroups []int,
) float64 {
	score := 0.0

	// Find longest consecutive prefix where pod has ALL full attention groups
	for i := 0; i < len(keys); i++ {
		entries := keyToPods[keys[i]]
		hasAllGroups := false

		for _, entry := range entries {
			if entry.PodIdentifier == podID {
				// Check if this entry has all required full attention groups
				hasAll := true
				for _, groupIdx := range fullGroups {
					if !containsGroup(entry.StoredGroups, groupIdx) {
						hasAll = false
						break
					}
				}
				if hasAll {
					hasAllGroups = true
					// Get tier weight
					weight := 1.0
					if s.MediumWeights != nil {
						if w, exists := s.MediumWeights[entry.DeviceTier]; exists {
							weight = w
						}
					}
					score += weight
				}
				break
			}
		}

		if !hasAllGroups {
			break // Prefix broken, stop counting
		}
	}

	return score
}

// scoreSWAForPod scores SWA groups using longest suffix matching from end.
// Matches vLLM's right-to-left search for sliding window attention.
// Returns weighted sum of matching blocks from end (up to window size).
func (s *HybridPrefixCacheScorer) scoreSWAForPod(
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
	podID string,
	swaGroups []int,
) float64 {
	if len(keys) == 0 || len(swaGroups) == 0 {
		return 0.0
	}

	// Find the minimum window size across all SWA groups
	minWindowBlocks := len(keys) // Default to all blocks
	for _, groupIdx := range swaGroups {
		if groupIdx < len(s.ModelConfig.AttentionGroups) {
			group := s.ModelConfig.AttentionGroups[groupIdx]
			blockSize := group.BlockSize
			if blockSize <= 0 {
				blockSize = s.ModelConfig.BlockSize
			}
			if blockSize <= 0 {
				continue
			}

			windowBlocks := (group.WindowSize + blockSize - 1) / blockSize
			if windowBlocks > 0 && windowBlocks < minWindowBlocks {
				minWindowBlocks = windowBlocks
			}
		}
	}

	score := 0.0
	numContiguous := 0

	// Search right-to-left from end (vLLM approach)
	for i := len(keys) - 1; i >= 0; i-- {
		entries := keyToPods[keys[i]]
		hasAllGroups := false

		for _, entry := range entries {
			if entry.PodIdentifier == podID {
				// Check if this entry has all required SWA groups
				hasAll := true
				for _, groupIdx := range swaGroups {
					if !containsGroup(entry.StoredGroups, groupIdx) {
						hasAll = false
						break
					}
				}
				if hasAll {
					hasAllGroups = true
					// Get tier weight
					weight := 1.0
					if s.MediumWeights != nil {
						if w, exists := s.MediumWeights[entry.DeviceTier]; exists {
							weight = w
						}
					}
					score += weight
					numContiguous++
				}
				break
			}
		}

		if !hasAllGroups {
			numContiguous = 0 // Reset, need contiguous blocks
			score = 0.0
		} else if numContiguous >= minWindowBlocks {
			break // Found enough blocks for window, stop
		}
	}

	return score
}

// scorePerPodHitLength calculates scores with per-pod hit length.
// Each pod is scored based on the minimum hit length across all groups for that pod.
// Score = avgTierWeight * (hitLength / totalRequestedBlocks)
// This prioritizes both cache completeness and tier quality.
func (s *HybridPrefixCacheScorer) scorePerPodHitLength(
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
) (map[string]float64, error) {
	totalRequestedBlocks := len(keys)

	// Step 1: Collect all pods
	allPods := make(map[string]struct{})
	for _, entries := range keyToPods {
		for _, entry := range entries {
			allPods[entry.PodIdentifier] = struct{}{}
		}
	}

	// Step 2: For each pod, calculate hit length per group
	podScores := make(map[string]float64)

	for podID := range allPods {
		// Calculate hit length for this pod across all groups
		// Track which group gave the minimum hit length to use its block size for scoring
		var minHitLength int
		var minBlockSize int

		for groupIdx, group := range s.ModelConfig.AttentionGroups {
			// Get block size for this group (fallback to model default if not set)
			blockSize := group.BlockSize
			if blockSize <= 0 {
				blockSize = s.ModelConfig.BlockSize
			}
			if blockSize <= 0 {
				return nil, fmt.Errorf("invalid block size for group %d: %d", groupIdx, blockSize)
			}

			// Find longest consecutive prefix for this pod in this group
			hitLength := s.findPodGroupCacheHit(keys, keyToPods, podID, groupIdx, &group, blockSize)

			// Track minimum hit length and its corresponding block size
			if groupIdx == 0 || hitLength < minHitLength {
				minHitLength = hitLength
				minBlockSize = blockSize
			}
		}

		// Skip pods with zero hit length
		if minHitLength == 0 {
			continue
		}

		// Calculate cumulative tier weight for blocks in hit length
		// Use the block size from the group that gave minimum hit length
		numBlocks := minHitLength / minBlockSize
		cumulativeTierWeight := 0.0

		for i := 0; i < numBlocks; i++ {
			entries := keyToPods[keys[i]]
			for _, entry := range entries {
				if entry.PodIdentifier == podID {
					weight := 1.0
					if s.MediumWeights != nil {
						if w, exists := s.MediumWeights[entry.DeviceTier]; exists {
							weight = w
						}
					}
					cumulativeTierWeight += weight
					break // Found this pod's entry for this block
				}
			}
		}

		// Normalize by total requested blocks to get completeness-weighted score
		// score = avgTierWeight * (hitLength / totalRequestedBlocks)
		//       = (sum(tierWeights) / hitLength) * (hitLength / totalRequestedBlocks)
		//       = sum(tierWeights) / totalRequestedBlocks
		if totalRequestedBlocks > 0 {
			podScores[podID] = cumulativeTierWeight / float64(totalRequestedBlocks)
		}
	}

	return podScores, nil
}

// findPodGroupCacheHit finds the longest consecutive prefix for a specific pod in a specific group.
func (s *HybridPrefixCacheScorer) findPodGroupCacheHit(
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
	podID string,
	groupIndex int,
	group *AttentionGroupConfig,
	blockSize int,
) int {
	// Apply window size constraint
	maxLength := len(keys) * blockSize
	if group.AttentionType != "full" && group.WindowSize > 0 && group.WindowSize < maxLength {
		maxLength = group.WindowSize
	}

	maxBlocks := maxLength / blockSize
	if maxBlocks > len(keys) {
		maxBlocks = len(keys)
	}

	// Find longest consecutive prefix where this specific pod has blocks with this group
	longestHit := 0

	for i := 0; i < maxBlocks; i++ {
		entries := keyToPods[keys[i]]
		found := false

		for _, entry := range entries {
			if entry.PodIdentifier == podID && containsGroup(entry.StoredGroups, groupIndex) {
				found = true
				break
			}
		}

		if found {
			longestHit = (i + 1) * blockSize
		} else {
			break // Consecutive chain broken
		}
	}

	return longestHit
}

// findLongestCacheHit implements the fixed-point algorithm to find the longest
// cache hit that works for all attention groups.
// DEPRECATED: This is kept for reference but no longer used in per-pod scoring.
func (s *HybridPrefixCacheScorer) findLongestCacheHit(
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
	maxHitLength int,
	blockSize int,
) int {
	numGroups := len(s.ModelConfig.AttentionGroups)
	hitLength := maxHitLength
	hitLengthByGroup := make([]int, numGroups)

	// Check if this is a simple hybrid (exactly 2 groups, one being full attention)
	// Simple hybrid converges in one iteration, matching vLLM's optimization.
	isSimpleHybrid := false
	if numGroups == 2 {
		for _, group := range s.ModelConfig.AttentionGroups {
			if group.AttentionType == "full" {
				isSimpleHybrid = true
				break
			}
		}
	}

	// Fixed-point iteration
	for {
		currHitLength := hitLength

		// Check each attention group's constraints
		for i, group := range s.ModelConfig.AttentionGroups {
			groupHitLength := s.findGroupCacheHit(
				keys, keyToPods, currHitLength, blockSize, i, &group)

			hitLengthByGroup[i] = groupHitLength

			// Update current hit length to minimum across all groups
			if groupHitLength < currHitLength {
				currHitLength = groupHitLength
			}
		}

		// Converged when hit length doesn't decrease
		if currHitLength >= hitLength {
			break
		}

		hitLength = currHitLength

		// Simple hybrid optimization: one iteration is enough
		if isSimpleHybrid {
			break
		}
	}

	return hitLength
}

// findGroupCacheHit finds the longest cache hit for a specific attention group.
// Only considers pods that have blocks cached for the given groupIndex.
func (s *HybridPrefixCacheScorer) findGroupCacheHit(
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
	maxLength int,
	blockSize int,
	groupIndex int,
	group *AttentionGroupConfig,
) int {
	// Apply window size constraint if specified
	// Full attention (AttentionType == "full") has no window constraint (WindowSize = 0)
	effectiveMaxLength := maxLength
	if group.AttentionType != "full" && group.WindowSize > 0 && group.WindowSize < maxLength {
		effectiveMaxLength = group.WindowSize
	}

	// Calculate max blocks based on effective length
	maxBlocks := effectiveMaxLength / blockSize
	if maxBlocks > len(keys) {
		maxBlocks = len(keys)
	}

	// Find longest consecutive prefix where at least one pod has all blocks
	// Similar to LongestPrefixScorer logic but respects group constraints
	if maxBlocks == 0 {
		return 0
	}

	// Track which pods have consecutive blocks
	activePods := make(map[string]struct{})

	// Initialize with pods that have the first block AND this group cached
	for _, entry := range keyToPods[keys[0]] {
		if containsGroup(entry.StoredGroups, groupIndex) {
			activePods[entry.PodIdentifier] = struct{}{}
		}
	}

	// Walk through blocks up to maxBlocks
	longestHit := 0
	for i := 0; i < maxBlocks; i++ {
		if len(activePods) == 0 {
			break
		}

		// Check current key - only include pods that have this group cached
		currentPods := make(map[string]struct{})
		for _, entry := range keyToPods[keys[i]] {
			if containsGroup(entry.StoredGroups, groupIndex) {
				currentPods[entry.PodIdentifier] = struct{}{}
			}
		}

		// Intersect: keep only pods that have this block
		for pod := range activePods {
			if _, exists := currentPods[pod]; !exists {
				delete(activePods, pod)
			}
		}

		if len(activePods) > 0 {
			longestHit = (i + 1) * blockSize
		} else {
			break
		}
	}

	return longestHit
}

// scoreByHitLength scores pods based on the converged hit length.
// Scores are normalized by the number of blocks to make them comparable across requests.
func (s *HybridPrefixCacheScorer) scoreByHitLength(
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
	hitLength int,
	blockSize int,
) (map[string]float64, error) {
	if hitLength == 0 {
		return make(map[string]float64), nil
	}

	numBlocks := hitLength / blockSize
	if numBlocks > len(keys) {
		numBlocks = len(keys)
	}

	podScores := make(map[string]float64)
	curWeights := make(map[string]float64)

	// Score pods that have all blocks up to hitLength
	for i := 0; i < numBlocks; i++ {
		clear(curWeights)
		fillMaxWeights(curWeights, keyToPods[keys[i]], s.MediumWeights)

		// Accumulate scores for pods that have this block
		for pod, weight := range curWeights {
			podScores[pod] += weight
		}
	}

	// Filter: only keep pods that have ALL blocks in the hit length
	activePods := make(map[string]struct{})
	for _, entry := range keyToPods[keys[0]] {
		activePods[entry.PodIdentifier] = struct{}{}
	}

	for i := 1; i < numBlocks; i++ {
		currentPods := make(map[string]struct{})
		for _, entry := range keyToPods[keys[i]] {
			currentPods[entry.PodIdentifier] = struct{}{}
		}

		for pod := range activePods {
			if _, exists := currentPods[pod]; !exists {
				delete(activePods, pod)
			}
		}
	}

	// Remove pods that don't have complete prefix
	for pod := range podScores {
		if _, exists := activePods[pod]; !exists {
			delete(podScores, pod)
		}
	}

	// Normalize scores by number of blocks (average weight per block)
	if numBlocks > 0 {
		for pod := range podScores {
			podScores[pod] /= float64(numBlocks)
		}
	}

	return podScores, nil
}

// scoreLongestPrefix is a fallback to simple longest prefix scoring when no attention groups configured.
func (s *HybridPrefixCacheScorer) scoreLongestPrefix(
	_ context.Context,
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
) (map[string]float64, error) {
	podScores := make(map[string]float64)
	curWeights := make(map[string]float64)

	fillMaxWeights(curWeights, keyToPods[keys[0]], s.MediumWeights)

	activePods := make(map[string]struct{}, len(curWeights))
	for pod, w := range curWeights {
		activePods[pod] = struct{}{}
		podScores[pod] = w
	}

	for i := 1; i < len(keys); i++ {
		if len(activePods) == 0 {
			break
		}

		clear(curWeights)
		fillMaxWeights(curWeights, keyToPods[keys[i]], s.MediumWeights)

		for pod := range activePods {
			if w, exists := curWeights[pod]; exists {
				podScores[pod] += w
			} else {
				delete(activePods, pod)
			}
		}
	}

	return podScores, nil
}

// containsGroup checks if a group ID is present in the StoredGroups slice.
// Returns true if groups is nil/empty (backwards compatibility) or if groupID is found.
// containsGroup checks if a group ID exists in the StoredGroups slice.
// For backward compatibility, returns true if groups is nil or empty.
func containsGroup(groups []int, groupID int) bool {
	// If StoredGroups is nil/empty, treat as "all groups" for backwards compatibility
	if len(groups) == 0 {
		return true
	}

	for _, g := range groups {
		if g == groupID {
			return true
		}
	}
	return false
}
