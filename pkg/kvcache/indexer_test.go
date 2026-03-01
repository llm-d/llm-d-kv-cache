//go:build embedded_tokenizers

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

package kvcache_test

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	"github.com/llm-d/llm-d-kv-cache/pkg/utils/logging"
)

const (
	testModelName = "test-model"
)

func newTestIndexer(t *testing.T) *kvcache.Indexer {
	t.Helper()
	ctx := logging.NewTestLoggerIntoContext(t.Context())

	modelDir := filepath.Join("..", "..", "tests", "e2e", "redis_mock", "testdata")
	localTokenizerConfig := tokenization.LocalTokenizerConfig{
		ModelTokenizerMap: map[string]string{
			testModelName: filepath.Join(modelDir, testModelName, "tokenizer.json"),
		},
	}

	config, err := kvcache.NewDefaultConfig()
	require.NoError(t, err)
	config.TokenizersPoolConfig = &tokenization.Config{
		ModelName:             testModelName,
		WorkersCount:          1,
		MinPrefixOverlapRatio: 0.8,
		LocalTokenizerConfig:  &localTokenizerConfig,
	}

	tokenProcessor := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())

	indexer, err := kvcache.NewKVCacheIndexer(ctx, config, tokenProcessor)
	require.NoError(t, err)
	go indexer.Run(ctx)

	return indexer
}

// TestComputeBlockKeys verifies that ComputeBlockKeys produces non-empty block
// keys for a valid prompt, and that calling it twice with the same input yields
// identical results.
func TestComputeBlockKeys(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	indexer := newTestIndexer(t)

	prompt := "One morning, when Gregor Samsa woke from troubled dreams, " +
		"he found himself transformed in his bed into a horrible vermin. " +
		"He lay on his armour-like back, and if he lifted his head a little he could see his brown belly, " +
		"slightly domed and divided by arches into stiff sections."

	blockKeys, err := indexer.ComputeBlockKeys(ctx, nil, prompt, testModelName)
	require.NoError(t, err)
	require.NotEmpty(t, blockKeys, "Expected non-empty block keys for a long prompt")

	// Calling again with the same input should produce identical results
	blockKeys2, err := indexer.ComputeBlockKeys(ctx, nil, prompt, testModelName)
	require.NoError(t, err)
	assert.Equal(t, blockKeys, blockKeys2, "ComputeBlockKeys should be deterministic")
}

// TestComputeBlockKeysConsistentWithGetPodScores verifies that ComputeBlockKeys
// produces block keys that match what GetPodScores uses internally. Since
// GetPodScores delegates to ComputeBlockKeys, we verify by adding known block
// keys to the index, then checking that GetPodScores finds them.
func TestComputeBlockKeysConsistentWithGetPodScores(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	indexer := newTestIndexer(t)

	prompt := "One morning, when Gregor Samsa woke from troubled dreams, " +
		"he found himself transformed in his bed into a horrible vermin. " +
		"He lay on his armour-like back, and if he lifted his head a little he could see his brown belly, " +
		"slightly domed and divided by arches into stiff sections."

	// Step 1: Compute block keys
	blockKeys, err := indexer.ComputeBlockKeys(ctx, nil, prompt, testModelName)
	require.NoError(t, err)
	require.NotEmpty(t, blockKeys)

	// Step 2: Add these block keys to the index for a test pod
	podAddr := "10.0.0.1:8080"
	kvBlockIndex := indexer.KVBlockIndex()
	for _, key := range blockKeys {
		err := kvBlockIndex.Add(ctx, []kvblock.BlockHash{key}, []kvblock.BlockHash{key}, []kvblock.PodEntry{
			{PodIdentifier: podAddr},
		})
		require.NoError(t, err)
	}

	// Step 3: GetPodScores should find the pod and give it a score
	scores, err := indexer.GetPodScores(ctx, nil, prompt, testModelName, []string{podAddr})
	require.NoError(t, err)
	require.Contains(t, scores, podAddr, "GetPodScores should find the pod using the same block keys")
	assert.Greater(t, scores[podAddr], 0.0, "Pod should have a positive score")
}

// TestComputeBlockKeysEmptyPrompt verifies that an empty or too-short prompt
// returns nil block keys without error.
func TestComputeBlockKeysEmptyPrompt(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	indexer := newTestIndexer(t)

	blockKeys, err := indexer.ComputeBlockKeys(ctx, nil, "", testModelName)
	require.NoError(t, err)
	assert.Nil(t, blockKeys, "Empty prompt should produce nil block keys")
}
