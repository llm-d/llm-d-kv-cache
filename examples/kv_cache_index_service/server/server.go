// Copyright 2025 The llm-d Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"context"
	"fmt"

	indexerpb "github.com/llm-d/llm-d-kv-cache/api/indexerpb"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
)

// IndexerService implements the IndexerServiceServer interface.
type IndexerService struct {
	indexerpb.UnimplementedIndexerServiceServer
	indexer      *kvcache.Indexer
	kvEventsPool *kvevents.Pool
}

// NewIndexerService creates a new IndexerService with the given indexer.
func NewIndexerService(pool *kvevents.Pool, indexer *kvcache.Indexer) *IndexerService {
	return &IndexerService{
		indexer:      indexer,
		kvEventsPool: pool,
	}
}

// GetPodScores implements the GetPodScores RPC method.
func (s *IndexerService) GetPodScores(ctx context.Context,
	req *indexerpb.GetPodScoresRequest,
) (*indexerpb.GetPodScoresResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	// Call the underlying indexer
	podScores, err := s.indexer.GetPodScores(ctx, nil, req.Prompt, req.ModelName,
		req.PodIdentifiers)
	if err != nil {
		return nil, fmt.Errorf("failed to get pod scores: %w", err)
	}

	// Convert map[string]float64 to []*indexerpb.PodScore. Scoring keys are
	// "pod-1" (non-DP) or "pod-1@dp0" (DP-aware); decomposition rules live in
	// kvcache.ParsePodScoringKey so the server and scorer cannot drift apart.
	scores := make([]*indexerpb.PodScore, 0, len(podScores))
	for scoringKey, score := range podScores {
		pod, dpRank := kvcache.ParsePodScoringKey(scoringKey)
		scores = append(scores, &indexerpb.PodScore{
			Pod:              pod,
			Score:            score,
			DataParallelRank: dpRank,
		})
	}

	return &indexerpb.GetPodScoresResponse{
		Scores: scores,
	}, nil
}
