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

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/telemetry"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"
	"github.com/llm-d/llm-d-kv-cache/pkg/utils/logging"
)

// Config holds the configuration for the Indexer module.
// The configuration cover the different components found in the Indexer
// module.
type Config struct {
	KVBlockIndexConfig  *kvblock.IndexConfig    `json:"kvBlockIndexConfig"`
	KVBlockScorerConfig *KVBlockScorerConfig    // not exported
	BackendConfigs      []*KVCacheBackendConfig `json:"kvCacheBackendConfigs"`

	// TokenizersPoolConfig configures the in-process tokenization pool.
	// Leaving it nil disables the pool; the prompt-string entry points then
	// return an error.
	//
	// Deprecated: tokenize externally and call Indexer.ScoreTokens.
	TokenizersPoolConfig *tokenization.Config `json:"tokenizersPoolConfig,omitempty"`
}

// NewDefaultConfig returns a default configuration for the Indexer module.
// TokenizersPoolConfig is left nil; populate it only if the deprecated
// prompt-string APIs are needed.
func NewDefaultConfig() (*Config, error) {
	return &Config{
		KVBlockIndexConfig:  kvblock.DefaultIndexConfig(),
		KVBlockScorerConfig: DefaultKVBlockScorerConfig(),
		BackendConfigs:      DefaultKVCacheBackendConfig(),
	}, nil
}

// Indexer is a concrete implementation of the KVCacheIndex interface.
type Indexer struct {
	config *Config

	tokenProcessor kvblock.TokenProcessor // turns tokens to kv block keys
	kvBlockIndex   kvblock.Index          // looks up pods for block keys
	kvBlockScorer  KVBlockScorer          // scores pods based on block hits

	tokenizersPool TokenizersPool
}

// NewKVCacheIndexer creates a KVCacheIndex given a Config. When
// config.TokenizersPoolConfig is nil, the indexer accepts only the tokens-in
// API (Indexer.ScoreTokens) and the prompt-string entry points return an error.
func NewKVCacheIndexer(ctx context.Context, config *Config, tokenProcessor kvblock.TokenProcessor) (*Indexer, error) {
	if config == nil {
		return nil, fmt.Errorf("config cannot be nil")
	}
	if tokenProcessor == nil {
		return nil, fmt.Errorf("tokenProcessor cannot be nil")
	}

	kvBlockIndex, err := kvblock.NewIndex(ctx, config.KVBlockIndexConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create RedisKVBlockIndexer: %w", err)
	}

	// Wrap index with tracing instrumentation.
	// When tracing is not configured, otel.Tracer() returns a no-op implementation.
	kvBlockIndex = kvblock.NewTracedIndex(kvBlockIndex)

	// override backend configs with the ones from the config, if the defaults are not used.
	config.KVBlockScorerConfig.BackendConfigs = config.BackendConfigs
	scorer, err := NewKVBlockScorer(config.KVBlockScorerConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create KVBlockScorer: %w", err)
	}

	// Wrap scorer with tracing instrumentation.
	// When tracing is not configured, otel.Tracer() returns a no-op implementation.
	scorer = NewTracedScorer(scorer)

	indexer := &Indexer{
		config:         config,
		tokenProcessor: tokenProcessor,
		kvBlockIndex:   kvBlockIndex,
		kvBlockScorer:  scorer,
	}

	if config.TokenizersPoolConfig != nil {
		tokenizersPool, err := tokenization.NewTokenizationPool(ctx, config.TokenizersPoolConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create tokenizers pool: %w", err)
		}
		indexer.tokenizersPool = tokenizersPool
	}

	return indexer, nil
}

// Run starts the indexer. Blocks until ctx is cancelled.
func (k *Indexer) Run(ctx context.Context) {
	if k.tokenizersPool == nil {
		<-ctx.Done()
		return
	}
	k.tokenizersPool.Run(ctx)
}

// KVBlockIndex returns the kvblock.Index used by the Indexer.
func (k *Indexer) KVBlockIndex() kvblock.Index {
	return k.kvBlockIndex
}

// ErrInternalTokenizationDisabled is returned by the deprecated prompt-string
// entry points when the indexer was constructed without TokenizersPoolConfig.
// Callers can inspect it via errors.Is to distinguish missing-pool from other
// failures.
var ErrInternalTokenizationDisabled = fmt.Errorf(
	"internal tokenization not configured: tokenize externally and call ScoreTokens / ComputeBlockKeysFromTokens")

// ComputeBlockKeys computes the KV-block keys for a given prompt and model name.
//
// Deprecated: use ComputeBlockKeysFromTokens.
func (k *Indexer) ComputeBlockKeys(ctx context.Context, renderReq *types.RenderChatRequest, prompt, modelName string,
) ([]kvblock.BlockHash, error) {
	if k.tokenizersPool == nil {
		return nil, ErrInternalTokenizationDisabled
	}

	// 1. tokenize prompt
	tokens, features := k.tokenizersPool.Tokenize(renderReq, prompt)

	// 2. Truncate prompt (if set in the request)
	if renderReq != nil && renderReq.TruncatePromptTokens != nil {
		limit := *renderReq.TruncatePromptTokens
		if limit > 0 && len(tokens) > limit {
			tokens = tokens[len(tokens)-limit:]
		}
	}

	// 3. Compute per-block extra features from multimodal metadata (if present).
	var extraFeatures []*kvblock.BlockExtraFeatures
	if features != nil {
		extraFeatures = kvblock.ComputeBlockExtraFeatures(
			features.MMHashes, features.MMPlaceholders,
			k.blockSize(), len(tokens))
	}

	return k.ComputeBlockKeysFromTokens(ctx, tokens, modelName, extraFeatures)
}

// ComputeBlockKeysFromTokens computes the KV-block keys for a pre-tokenized
// prompt. Callers tokenize and truncate externally. extraFeatures provides
// per-block multimodal data that taints the hash; nil means text-only.
func (k *Indexer) ComputeBlockKeysFromTokens(ctx context.Context, tokens []uint32, modelName string,
	extraFeatures []*kvblock.BlockExtraFeatures,
) ([]kvblock.BlockHash, error) {
	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("kvcache.ComputeBlockKeysFromTokens")

	blockKeys, err := k.tokenProcessor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, modelName, extraFeatures)
	if err != nil {
		traceLogger.Error(err, "blockKey conversion failed")
		return nil, fmt.Errorf("blockKey conversion failed: %w", err)
	}
	if len(blockKeys) == 0 {
		traceLogger.Info("no block keys found")
		return nil, nil
	}
	traceLogger.Info("computed block keys", "tokens", tokens, "block-keys", blockKeys)

	return blockKeys, nil
}

// GetPodScores retrieves the pod scores for a given prompt and model name.
// A pod identifier should be its address. An empty podIdentifiers set means
// all pods are considered.
//
// Deprecated: use ScoreTokens.
func (k *Indexer) GetPodScores(ctx context.Context, renderReq *types.RenderChatRequest, prompt, modelName string,
	podIdentifiers []string,
) (map[string]float64, error) {
	if k.tokenizersPool == nil {
		return nil, ErrInternalTokenizationDisabled
	}

	// 1. tokenize prompt
	tokens, features := k.tokenizersPool.Tokenize(renderReq, prompt)

	// 2. Truncate prompt (if set in the request)
	if renderReq != nil && renderReq.TruncatePromptTokens != nil {
		limit := *renderReq.TruncatePromptTokens
		if limit > 0 && len(tokens) > limit {
			tokens = tokens[len(tokens)-limit:]
		}
	}

	// 3. Compute per-block extra features from multimodal metadata (if present).
	var extraFeatures []*kvblock.BlockExtraFeatures
	if features != nil {
		extraFeatures = kvblock.ComputeBlockExtraFeatures(
			features.MMHashes, features.MMPlaceholders,
			k.blockSize(), len(tokens))
	}

	return k.ScoreTokens(ctx, tokens, modelName, podIdentifiers, extraFeatures)
}

// ScoreTokens computes pod scores for the given tokens and model.
// It converts tokens into KV block keys, looks up which pods hold
// matching blocks in the index, and scores each pod based on cache hits.
//
// extraFeatures provides per-block multimodal data that taints the hash;
// nil means text-only. podIdentifiers limits scoring to the given pod addresses.
// If empty, all pods are considered.
func (k *Indexer) ScoreTokens(
	ctx context.Context,
	tokens []uint32,
	modelName string,
	podIdentifiers []string,
	extraFeatures []*kvblock.BlockExtraFeatures,
) (map[string]float64, error) {
	tracer := otel.Tracer(telemetry.InstrumentationName)
	ctx, span := tracer.Start(ctx, "llm_d.kv_cache.score_tokens",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("kvcache.ScoreTokens")

	blockKeys, err := k.tokenProcessor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, modelName, extraFeatures)
	if err != nil {
		return nil, fmt.Errorf("blockKey conversion failed: %w", err)
	}

	span.SetAttributes(
		attribute.String("gen_ai.request.model", modelName),
		attribute.Int("llm_d.kv_cache.pod_count", len(podIdentifiers)),
		attribute.Int("llm_d.kv_cache.token_count", len(tokens)),
		attribute.Int("llm_d.kv_cache.block_keys.count", len(blockKeys)),
	)

	if len(blockKeys) == 0 {
		traceLogger.Info("no block keys found, returning empty scores")
		//nolint:nilnil // no need to return an error
		return nil, nil
	}
	traceLogger.Info("found tokens", "tokens", tokens, "block-keys", blockKeys)

	keyToPods, err := k.kvBlockIndex.Lookup(ctx, blockKeys, sets.New(podIdentifiers...))
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
		return nil, fmt.Errorf("failed to query kvblock indexer: %w", err)
	}
	traceLogger.Info("found block keys", "block-keys", blockKeys,
		"pods", podsPerKeyPrintHelper(keyToPods))

	// Calculate block-level hit ratio (blocks found / blocks requested).
	blocksFound := 0
	for _, pods := range keyToPods {
		if len(pods) > 0 {
			blocksFound++
		}
	}
	blockHitRatio := 0.0
	if len(blockKeys) > 0 {
		blockHitRatio = float64(blocksFound) / float64(len(blockKeys))
	}
	span.SetAttributes(
		attribute.Float64("llm_d.kv_cache.block_hit_ratio", blockHitRatio),
		attribute.Int("llm_d.kv_cache.blocks_found", blocksFound),
	)

	podScores, err := k.kvBlockScorer.Score(ctx, blockKeys, keyToPods)
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
		return nil, fmt.Errorf("failed to query kvblock scorer: %w", err)
	}

	return podScores, nil
}

// podsPerKeyPrintHelper formats a map of keys to pod entries for printing.
func podsPerKeyPrintHelper(ks map[kvblock.BlockHash][]kvblock.PodEntry) string {
	flattened := ""
	for k, v := range ks {
		entries := make([]string, len(v))
		for i, entry := range v {
			entries[i] = entry.String()
		}
		flattened += fmt.Sprintf("%s: %v\n", k.String(), entries)
	}

	return flattened
}

// SetTokenizer overrides the in-process tokenizer. No-op when the pool is
// disabled.
//
// Deprecated: tied to the in-process tokenization pool.
func (k *Indexer) SetTokenizer(tokenizer tokenization.Tokenizer, modelName string) {
	if k.tokenizersPool == nil {
		return
	}
	k.tokenizersPool.SetTokenizer(tokenizer, modelName)
}

// blockSize returns the block size from the injected token processor.
func (k *Indexer) blockSize() int {
	return k.tokenProcessor.BlockSize()
}
