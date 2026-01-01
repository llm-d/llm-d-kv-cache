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

//nolint:testpackage // allow tests to run in the same package
package e2e

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/go-logr/logr/testr"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	preprocessing "github.com/llm-d/llm-d-kv-cache/pkg/preprocessing/chat_completions"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	"github.com/llm-d/llm-d-kv-cache/pkg/utils"
	"github.com/stretchr/testify/suite"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	defaultModelName = "google-bert/bert-base-uncased"
)

// KVCacheSuite defines a testify test suite for end-to-end testing of the KVCache indexer.
// It uses a mock Redis server (miniredis) and a tokenizer to simulate and verify cache behavior.
type KVCacheSuite struct {
	suite.Suite

	ctx    context.Context
	cancel context.CancelFunc

	tokenizer       tokenization.Tokenizer
	tokensProcessor kvblock.TokenProcessor
	config          *kvcache.Config

	kvBlockIndex kvblock.Index
	indexer      *kvcache.Indexer // TODO: test for all index backends

	Pod1IP string
}

// SetupTest initializes the mock Redis, tokenizer, config, and starts the indexer before each test.
func (s *KVCacheSuite) SetupTest() {
	// Initialize controller-runtime logger with test logger
	// This will display logs in test output with -v flag
	testLogger := testr.New(s.T())
	log.SetLogger(testLogger)

	s.ctx, s.cancel = context.WithCancel(context.Background())
	s.ctx = log.IntoContext(s.ctx, testLogger)

	var err error
	s.Require().NoError(err)

	s.config, err = kvcache.NewDefaultConfig()
	s.Require().NoError(err)

	s.config.PrefixStoreConfig.BlockSize = 4
	s.config.TokenProcessorConfig.BlockSize = 4

	// Configure the indexer's tokenization pool to support local models
	// This is needed because GetPodScores uses the indexer's internal pool for tokenization
	testDataPath, err := filepath.Abs("testdata")
	s.Require().NoError(err)

	s.config.TokenizersPoolConfig.LocalTokenizerConfig.AutoDiscoveryDir = testDataPath

	localTokenizer, err := tokenization.NewCachedLocalTokenizer(s.config.TokenizersPoolConfig.LocalTokenizerConfig)
	s.Require().NoError(err)

	hfTokenizer, err := tokenization.NewCachedHFTokenizer(s.config.TokenizersPoolConfig.HFTokenizerConfig)
	s.Require().NoError(err)

	// Use composite tokenizer: try local first, then fall back to HF
	s.tokenizer = &tokenization.CompositeTokenizer{
		Tokenizers: []tokenization.Tokenizer{localTokenizer, hfTokenizer},
	}

	s.tokensProcessor = kvblock.NewChunkedTokenDatabase(s.config.TokenProcessorConfig)

	s.Pod1IP = "10.0.0.1"

	s.indexer, err = kvcache.NewKVCacheIndexer(s.ctx, s.config)
	s.kvBlockIndex = s.indexer.KVBlockIndex()
	s.Require().NoError(err)

	go s.indexer.Run(s.ctx)
}

// promptToEngineAndRequestKeys tokenizes a prompt and returns its corresponding KV block keys.
// If tokenizer is provided, it will be used instead of the suite's default tokenizer.
//
//nolint:nonamedreturns // named returns keep gocritic unnamedResult satisfied while allowing compact return
func (s *KVCacheSuite) promptToEngineAndRequestKeys(
	prompt, model string,
) (engineKeys, requestKeys []kvblock.Key) {
	tokens, _, err := s.tokenizer.Encode(&preprocessing.EncodeRequest{
		ChatTemplateRequest: preprocessing.ChatTemplateRequest{Model: model},
		Text:                prompt,
		AddSpecialTokens:    true,
	})
	s.Require().NoError(err)

	requestKeys = s.tokensProcessor.TokensToKVBlockKeys(nil, tokens, model)
	s.Require().NotEmpty(requestKeys)

	engineKeys = s.tokensProcessor.TokensToKVBlockKeys(&kvblock.Key{ModelName: model, ChunkHash: 1}, tokens, model)
	s.Require().NotEmpty(engineKeys)

	return engineKeys, requestKeys
}

func (s *KVCacheSuite) addEntriesToIndex(engineKeys, requestKeys []kvblock.Key, podList []string) {
	s.Require().NotEmpty(engineKeys)
	s.Require().NotEmpty(requestKeys)

	// Add entries to the indexer
	err := s.kvBlockIndex.Add(s.ctx, engineKeys, requestKeys, utils.SliceMap(podList, func(pod string) kvblock.PodEntry {
		return kvblock.PodEntry{
			PodIdentifier: pod,
			DeviceTier:    "gpu",
		}
	}))
	s.Require().NoError(err)
}

// TestKVCacheSuite runs the KVCacheSuite using testify's suite runner.
func TestKVCacheSuite(t *testing.T) {
	suite.Run(t, new(KVCacheSuite))
}
