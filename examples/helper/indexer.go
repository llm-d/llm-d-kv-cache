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

package helper

import (
	"context"
	"os"
	"strings"

	"github.com/llm-d/llm-d-kv-cache/examples/testdata"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	// envTokenizerEndpoint overrides the UDS tokenizer socket path or TCP address.
	// Use a path (e.g. /tmp/tokenizer/tokenizer-uds.socket) for UDS mode (default),
	// or host:port (e.g. localhost:50051) for TCP mode (useful when running the
	// tokenizer as a Docker container).
	envTokenizerEndpoint = "TOKENIZER_ENDPOINT"
)

func getKVCacheIndexerConfig() (*kvcache.Config, error) {
	config, err := kvcache.NewDefaultConfig()
	if err != nil {
		return nil, err
	}

	config.TokenizersPoolConfig.ModelName = testdata.ModelName

	if endpoint := os.Getenv(envTokenizerEndpoint); endpoint != "" {
		config.TokenizersPoolConfig.UdsTokenizerConfig = &tokenization.UdsTokenizerConfig{
			SocketFile: endpoint,
			UseTCP:     !strings.HasPrefix(endpoint, "/"),
		}
	}

	return config, nil
}

func getTokenProcessorConfig() *kvblock.TokenProcessorConfig {
	return &kvblock.TokenProcessorConfig{
		BlockSize: 256,
	}
}

func SetupKVCacheIndexer(ctx context.Context) (*kvcache.Indexer, error) {
	logger := log.FromContext(ctx)

	cfg, err := getKVCacheIndexerConfig()
	if err != nil {
		return nil, err
	}

	tokenProcessor, err := kvblock.NewChunkedTokenDatabase(getTokenProcessorConfig())
	if err != nil {
		return nil, err
	}

	kvCacheIndexer, err := kvcache.NewKVCacheIndexer(ctx, cfg, tokenProcessor)
	if err != nil {
		return nil, err
	}

	logger.Info("Created Indexer")

	go kvCacheIndexer.Run(ctx)
	logger.Info("Started Indexer", "model", testdata.ModelName)

	return kvCacheIndexer, nil
}
