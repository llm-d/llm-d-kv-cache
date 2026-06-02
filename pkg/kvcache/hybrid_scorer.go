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
	"slices"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// HybridPrefixCacheScorer scores pods for HMA models using per-group boundary
// evaluation. Full attention acts as a kill switch (must be consecutive from
// block 0), SWA groups track contiguous runs against their window threshold.
// Final score = min(fullLastSeq, min(swaLastSeqs)) + 1.
//
// When attentionInfo is nil it falls back to longest prefix scoring.
type HybridPrefixCacheScorer struct {
	MediumWeights map[string]float64
	DefaultScorer *LongestPrefixScorer
}

func (s *HybridPrefixCacheScorer) Strategy() KVScoringStrategy {
	return HybridPrefixMatch
}

func (s *HybridPrefixCacheScorer) Score(
	ctx context.Context,
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
	attentionInfo *kvblock.AttentionInfo,
) (map[string]float64, error) {
	if len(keys) == 0 {
		return make(map[string]float64), nil
	}

	logger := log.FromContext(ctx)

	if attentionInfo == nil {
		logger.V(1).Info("no AttentionInfo, using LongestPrefix scorer")
		return s.DefaultScorer.Score(ctx, keys, keyToPods, nil)
	}

	logger.V(1).Info("using HybridPrefix scorer")
	return s.hybridScore(keys, keyToPods, attentionInfo), nil
}

type podHMAState struct {
	fullActive  bool
	fullLastSeq int
	swaCount    []int
	swaLastSeq  []int // -1 = threshold never met
}

// collectGroups builds a map of podIdentifier → set of GroupIDs present at a block.
func collectGroups(entries []kvblock.PodEntry) map[string][]int {
	groups := make(map[string][]int)
	for _, entry := range entries {
		if entry.GroupID >= 0 {
			gids := groups[entry.PodIdentifier]
			if !slices.Contains(gids, entry.GroupID) {
				groups[entry.PodIdentifier] = append(gids, entry.GroupID)
			}
		}
	}
	return groups
}

func (s *HybridPrefixCacheScorer) hybridScore(
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
	info *kvblock.AttentionInfo,
) map[string]float64 {
	numSWA := len(info.SWAGroupIDs)
	states := make(map[string]*podHMAState)

	block0Groups := collectGroups(keyToPods[keys[0]])
	for podID, gids := range block0Groups {
		if !slices.Contains(gids, info.FullGroupID) {
			continue
		}
		st := &podHMAState{
			fullActive:  true,
			fullLastSeq: 0,
			swaCount:    make([]int, numSWA),
			swaLastSeq:  make([]int, numSWA),
		}
		for i, swaGID := range info.SWAGroupIDs {
			st.swaLastSeq[i] = -1
			if slices.Contains(gids, swaGID) {
				st.swaCount[i] = 1
				if st.swaCount[i] >= info.SWAWindowBlocks[i] {
					st.swaLastSeq[i] = 0
				}
			}
		}
		states[podID] = st
	}

	for blockIdx := 1; blockIdx < len(keys); blockIdx++ {
		if len(states) == 0 {
			break
		}

		podGroups := collectGroups(keyToPods[keys[blockIdx]])

		for podID, st := range states {
			gids := podGroups[podID]
			present := len(gids) > 0

			if st.fullActive {
				if present && slices.Contains(gids, info.FullGroupID) {
					st.fullLastSeq = blockIdx
				} else {
					st.fullActive = false
				}
			}

			for i, swaGID := range info.SWAGroupIDs {
				if present && slices.Contains(gids, swaGID) {
					st.swaCount[i]++
					if st.swaCount[i] >= info.SWAWindowBlocks[i] {
						st.swaLastSeq[i] = blockIdx
					}
				} else {
					st.swaCount[i] = 0
				}
			}
		}
	}

	scores := make(map[string]float64)
	for podID, st := range states {
		checkpoint := st.fullLastSeq
		allMet := true
		for i := range numSWA {
			if st.swaLastSeq[i] < 0 {
				allMet = false
				break
			}
			if st.swaLastSeq[i] < checkpoint {
				checkpoint = st.swaLastSeq[i]
			}
		}
		if !allMet {
			continue
		}
		scores[podID] = float64(checkpoint + 1)
	}

	return scores
}
