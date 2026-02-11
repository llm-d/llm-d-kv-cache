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
	_ "embed"
	"fmt"
	"os"
	"time"

	"github.com/llm-d/llm-d-kv-cache/pkg/utils"
	"github.com/redis/go-redis/v9"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"

	"github.com/llm-d/llm-d-kv-cache/examples/testdata"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
)

const (
	defaultModelName = testdata.ModelName

	envRedisAddr = "REDIS_ADDR"
	envHFToken   = "HF_TOKEN"
	envModelName = "MODEL_NAME"
)

func getKVCacheIndexerConfig() (*kvcache.Config, error) {
	config, err := kvcache.NewDefaultConfig()
	if err != nil {
		return nil, err
	}

	config.TokenizersPoolConfig.ModelName = getModelName()

	huggingFaceToken := os.Getenv(envHFToken)
	if huggingFaceToken != "" && config.TokenizersPoolConfig.HFTokenizerConfig != nil {
		config.TokenizersPoolConfig.HFTokenizerConfig.HuggingFaceToken = huggingFaceToken
	}

	redisAddr := os.Getenv(envRedisAddr)
	if redisAddr != "" {
		redisOpt, err := redis.ParseURL(redisAddr)
		if err != nil {
			return nil, fmt.Errorf("failed to parse redis host: %w", err)
		}

		config.KVBlockIndexConfig.RedisConfig.Address = redisOpt.Addr
	} // Otherwise defaults to in-memory indexer

	return config, nil
}

func getModelName() string {
	modelName := os.Getenv(envModelName)
	if modelName != "" {
		return modelName
	}

	return defaultModelName
}

func main() {
	baseLogger := zap.New(zap.UseDevMode(true))
	log.SetLogger(baseLogger)

	ctx := log.IntoContext(context.Background(), baseLogger)
	logger := log.FromContext(ctx)

	kvCacheIndexer, err := setupKVCacheIndexer(ctx)
	if err != nil {
		logger.Error(err, "failed to set up KVCacheIndexer")
		os.Exit(1)
	}

	if err := runPrompts(ctx, kvCacheIndexer); err != nil {
		logger.Error(err, "failed to run prompts")
		os.Exit(1)
	}
}

func setupKVCacheIndexer(ctx context.Context) (*kvcache.Indexer, error) {
	logger := log.FromContext(ctx)

	config, err := getKVCacheIndexerConfig()
	if err != nil {
		return nil, err
	}

	config.TokenizersPoolConfig.ModelName = testdata.ModelName

	kvCacheIndexer, err := kvcache.NewKVCacheIndexer(ctx, config,
		kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{
			BlockSize: 256,
		}),
	)
	if err != nil {
		return nil, err
	}

	logger.Info("Created Indexer")

	go kvCacheIndexer.Run(ctx)
	modelName := getModelName()
	logger.Info("Started Indexer", "model", modelName)

	return kvCacheIndexer, nil
}

func runPrompts(ctx context.Context, kvCacheIndexer *kvcache.Indexer) error {
	logger := log.FromContext(ctx)

	modelName := getModelName()
	logger.Info("Started Indexer", "model", modelName)

	// Get pods for the prompt
	pods, err := kvCacheIndexer.GetPodScores(ctx, testdata.RenderReq, testdata.Prompt, modelName, nil)
	if err != nil {
		return err
	}

	// Print the pods - should be empty because no tokenization
	logger.Info("Got pods", "pods", pods)

	// Add entries in kvblock.Index manually
	engineKeys := utils.SliceMap(testdata.PromptHashes, func(h uint64) kvblock.BlockHash {
		return kvblock.BlockHash(h)
	})
	// For this simple example, requestKeys == engineKeys
	requestKeys := engineKeys

	if err := kvCacheIndexer.KVBlockIndex().Add(ctx, engineKeys, requestKeys,
		[]kvblock.PodEntry{{PodIdentifier: "pod1", DeviceTier: "gpu"}}); err != nil {
		return err
	}

	// Sleep 3 secs
	time.Sleep(3 * time.Second)

	// Get pods for the prompt
	pods, err = kvCacheIndexer.GetPodScores(ctx, testdata.RenderReq, testdata.Prompt, modelName, nil)
	if err != nil {
		return err
	}

	// Print the pods - should be empty because no tokenization
	logger.Info("Got pods", "pods", pods)
	return nil
}
