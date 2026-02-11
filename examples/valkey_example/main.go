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

package main

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/llm-d/llm-d-kv-cache/examples/testdata"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/utils"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
)

const (
	envValkeyAddr       = "VALKEY_ADDR"
	envValkeyEnableRDMA = "VALKEY_ENABLE_RDMA"
	envHFToken          = "HF_TOKEN"
)

func main() {
	baseLogger := zap.New(zap.UseDevMode(true))
	log.SetLogger(baseLogger)

	ctx := log.IntoContext(context.Background(), baseLogger)
	logger := log.FromContext(ctx)

	// Create KV-Cache Manager configuration with Valkey backend
	config, err := createValkeyConfig()
	if err != nil {
		logger.Error(err, "failed to create valkey config")
		os.Exit(1)
	}

	logger.Info("Initializing KV-Cache Manager with Valkey backend",
		"valkeyAddr", config.KVBlockIndexConfig.ValkeyConfig.Address,
		"rdmaEnabled", config.KVBlockIndexConfig.ValkeyConfig.EnableRDMA)

	// Initialize the KV-Cache indexer
	indexer, err := kvcache.NewKVCacheIndexer(ctx, config,
		kvblock.NewChunkedTokenDatabase(createTokenProcessorConfig()))
	if err != nil {
		logger.Error(err, "failed to create KV-Cache indexer")
		os.Exit(1)
	}

	// Start the indexer in the background
	go indexer.Run(ctx)

	// Demonstrate Valkey operations
	if err := demonstrateValkeyOperations(ctx, indexer); err != nil {
		logger.Error(err, "Valkey operations failed")
		os.Exit(1)
	}

	logger.Info("Valkey example completed successfully")
}

func createValkeyConfig() (*kvcache.Config, error) {
	config, err := kvcache.NewDefaultConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to create default config: %w", err)
	}

	config.TokenizersPoolConfig.ModelName = testdata.ModelName

	// Configure Valkey backend
	valkeyAddr := os.Getenv(envValkeyAddr)
	if valkeyAddr == "" {
		valkeyAddr = "valkey://127.0.0.1:6379"
	}

	enableRDMA := false
	if rdmaEnv := os.Getenv(envValkeyEnableRDMA); rdmaEnv == "true" {
		enableRDMA = true
	}

	// Set up Valkey configuration
	config.KVBlockIndexConfig = &kvblock.IndexConfig{
		ValkeyConfig: &kvblock.RedisIndexConfig{
			Address:     valkeyAddr,
			BackendType: "valkey",
			EnableRDMA:  enableRDMA,
		},
		EnableMetrics:          true,
		MetricsLoggingInterval: 30 * time.Second,
	}

	// Configure tokenizer
	if hfToken := os.Getenv(envHFToken); hfToken != "" {
		config.TokenizersPoolConfig.HFTokenizerConfig.HuggingFaceToken = hfToken
	}

	config.TokenizersPoolConfig.ModelName = testdata.ModelName
	return config, nil
}

func createTokenProcessorConfig() *kvblock.TokenProcessorConfig {
	// Set a reasonable block size for demonstration
	return &kvblock.TokenProcessorConfig{
		BlockSize: 128,
	}
}

func demonstrateValkeyOperations(ctx context.Context, indexer *kvcache.Indexer) error {
	logger := log.FromContext(ctx).WithName("valkey-demo")

	modelName := testdata.ModelName
	prompt := testdata.Prompt

	podEntries := []kvblock.PodEntry{
		{PodIdentifier: "demo-pod-1", DeviceTier: "gpu"},
		{PodIdentifier: "demo-pod-2", DeviceTier: "gpu"},
	}

	logger.Info("Processing testdata prompt", "model", modelName, "promptLength", len(prompt))

	// First, let's demonstrate basic scoring without any cache entries
	scores, err := indexer.GetPodScores(ctx, nil, prompt, modelName, []string{"demo-pod-1", "demo-pod-2"})
	if err != nil {
		return fmt.Errorf("failed to get pod scores: %w", err)
	}

	logger.Info("Initial cache scores (should be empty)", "scores", scores)

	// Now let's manually add some cache entries using the pre-calculated hashes from testdata
	logger.Info("Adding cache entries manually to demonstrate Valkey backend")

	// Use the pre-calculated hashes from testdata that match the prompt
	promptKeys := utils.SliceMap(testdata.PromptHashes, func(h uint64) kvblock.BlockHash {
		return kvblock.BlockHash(h)
	})

	// In this example, requestKeys are identical to engineKeys (promptKeys)
	requestKeys := promptKeys

	err = indexer.KVBlockIndex().Add(ctx, promptKeys, requestKeys, podEntries)
	if err != nil {
		return fmt.Errorf("failed to add cache entries: %w", err)
	}

	logger.Info("Added cache entries", "keys", len(promptKeys), "pods", len(podEntries))

	// Query for cache scores again
	scores, err = indexer.GetPodScores(ctx, nil, prompt, modelName, []string{"demo-pod-1", "demo-pod-2"})
	if err != nil {
		return fmt.Errorf("failed to get pod scores after adding entries: %w", err)
	}

	logger.Info("Cache scores after adding entries", "scores", scores)

	// Demonstrate lookup functionality
	logger.Info("Demonstrating cache lookup via Valkey backend")
	lookupResults, err := indexer.KVBlockIndex().Lookup(ctx, promptKeys, nil)
	if err != nil {
		return fmt.Errorf("failed to lookup cache entries: %w", err)
	}

	logger.Info("Cache lookup results", "keysFound", len(lookupResults))
	for key, pods := range lookupResults {
		logger.Info("Key found", "key", key.String(), "pods", pods)
	}

	// Demonstrate eviction
	logger.Info("Demonstrating cache eviction")
	err = indexer.KVBlockIndex().Evict(ctx, promptKeys[0], podEntries[:1])
	if err != nil {
		return fmt.Errorf("failed to evict cache entry: %w", err)
	}

	// Lookup again after eviction
	lookupAfterEvict, err := indexer.KVBlockIndex().Lookup(ctx, promptKeys, nil)
	if err != nil {
		return fmt.Errorf("failed to lookup after eviction: %w", err)
	}

	logger.Info("Cache lookup after eviction", "keysFound", len(lookupAfterEvict))

	// Final score check to see the difference
	finalScores, err := indexer.GetPodScores(ctx, nil, prompt, modelName, []string{"demo-pod-1", "demo-pod-2"})
	if err != nil {
		return fmt.Errorf("failed to get final pod scores: %w", err)
	}

	logger.Info("Final cache scores", "scores", finalScores)

	return nil
}
