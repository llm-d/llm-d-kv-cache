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
	"fmt"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"k8s.io/apimachinery/pkg/util/sets"
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
	Score(keys []kvblock.BlockHash, keyToPods map[kvblock.BlockHash][]kvblock.PodEntry) (map[string]float64, error)
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

// getMaxWeight returns the maximum weight for a given pod across all device tiers.
func getMaxWeight(entries []kvblock.PodEntry, podID string, mediumWeights map[string]float64) float64 {
	maxWeight := 0.0
	for _, entry := range entries {
		if entry.PodIdentifier == podID {
			weight := 1.0
			if mediumWeights != nil {
				if w, exists := mediumWeights[entry.DeviceTier]; exists {
					weight = w
				}
			}
			if weight > maxWeight {
				maxWeight = weight
			}
		}
	}
	return maxWeight
}

// Score implements the longest prefix scoring logic with weighted sum based on BackendConfig.
func (s *LongestPrefixScorer) Score(
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
) (map[string]float64, error) {
	podScores := make(map[string]float64)

	if len(keys) == 0 {
		return make(map[string]float64), nil
	}

	podsForFirstKey := keyToPods[keys[0]]

	activePods := sets.NewString()
	for _, pod := range podsForFirstKey {
		activePods.Insert(pod.PodIdentifier)
	}

	// pods not in the first key will retain the default score of 0.
	for pod := range activePods {
		podScores[pod] = getMaxWeight(podsForFirstKey, pod, s.MediumWeights)
	}

	for i := 1; i < len(keys); i++ {
		if activePods.Len() == 0 {
			break
		}

		podsForKey := keyToPods[keys[i]]
		currentPodsSet := sets.NewString()
		for _, pod := range podsForKey {
			currentPodsSet.Insert(pod.PodIdentifier)
		}

		// update scores and active pods to the intersection
		activePods = activePods.Intersection(currentPodsSet)
		for pod := range activePods {
			// increment score for each pod in the intersection
			podScores[pod] += getMaxWeight(podsForKey, pod, s.MediumWeights)
		}
	}

	// Return the map containing the final score for each pod encountered.
	return podScores, nil
}
