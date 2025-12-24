//go:build exclude

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

package kv_cache_aware_scorer

import (
	"context"
	"fmt"
	"os"

	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache"
)

const (
	kvCacheAwareScorerName = "kvcache-aware-scorer"

	kvCacheRedisEnvVar     = "KVCACHE_INDEXER_REDIS_ADDR"
	huggingFaceTokenEnvVar = "HF_TOKEN"
)

// KVCacheAwareScorer uses the KVCacheIndexer to score pods based on KVCache
// awareness.
type KVCacheAwareScorer struct {
	kvCacheIndexer *kvcache.Indexer
}

// NewKVCacheAwareScorer creates a new KVCacheAwareScorer instance.
// It initializes the KVCacheIndexer from environment variables.
//
// If the environment variables are not set, or if the indexer
// fails to initialize, an error is returned.
func NewKVCacheAwareScorer(ctx context.Context) (plugins.Scorer, error) {
	config, err := kvcache.NewDefaultConfig()
	if err != nil {
		return nil, err
	}

	modelName := os.Getenv("MODEL_NAME")
	if modelName != "" {
		config.TokenizersPoolConfig.ModelName = modelName
	} else {
		return nil, fmt.Errorf("environment variable MODEL_NAME is not set")
	}

	redisAddr := os.Getenv(kvCacheRedisEnvVar)
	if redisAddr != "" {
		config.KVBlockIndexerConfig.RedisAddr = redisAddr
	} else {
		return nil, fmt.Errorf("environment variable %s is not set", kvCacheRedisEnvVar)
	}

	hfToken := os.Getenv(huggingFaceTokenEnvVar)
	if hfToken != "" {
		config.TokenizersPoolConfig.HuggingFaceToken = hfToken
	} else {
		return nil, fmt.Errorf("environment variable %s is not set", huggingFaceTokenEnvVar)
	}

	kvCacheIndexer, err := kvcache.NewKVCacheIndexer(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create KVCacheIndexer: %w", err)
	}

	go kvCacheIndexer.Run(ctx)

	return &KVCacheAwareScorer{
		kvCacheIndexer: kvCacheIndexer,
	}, nil
}

// Name returns the name of the scorer.
func (s *KVCacheAwareScorer) Name() string {
	return kvCacheAwareScorerName
}

// Score scores the provided pod based on the KVCache index state.
// The returned scores are normalized to a range of 0-1.
func (s *KVCacheAwareScorer) Score(ctx *types.SchedulingContext, pods []types.Pod) map[types.Pod]float64 {
	loggerDebug := log.FromContext(ctx).WithName(kvCacheAwareScorerName).V(logutil.DEBUG)
	if ctx.Req == nil {
		loggerDebug.Info("Request is nil, skipping scoring")
		return nil
	}

	scores, err := s.kvCacheIndexer.GetPodScores(ctx.Context, ctx.Req.Prompt, ctx.Req.Model, nil)
	if err != nil {
		loggerDebug.Error(err, "Failed to get pod scores")
		return nil
	}
	loggerDebug.Info("Got pod scores", "scores", scores)

	podToKey := func(pod types.Pod) (string, bool) {
		metricsPod := pod.GetPod()
		if metricsPod == nil {
			return "", false
		}

		return metricsPod.Address, true
	}

	return indexedScoresToNormalizedScoredPods(pods, podToKey, scores) // normalize scores
}
