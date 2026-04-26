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
	"net"
	"os"

	"github.com/llm-d/llm-d-kv-cache/examples/testdata"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	// EnvTokenizerEndpoint is the env var for the UDS tokenizer socket path or TCP address.
	// Use a path (e.g. /tmp/tokenizer/tokenizer-uds.socket) for UDS mode,
	// or host:port (e.g. localhost:50051) for TCP mode.
	EnvTokenizerEndpoint = "TOKENIZER_ENDPOINT" //nolint:gosec // env var name, not a credential
)

func isTCPAddr(s string) bool {
	host, port, err := net.SplitHostPort(s)
	return err == nil && host != "" && port != ""
}

// ConfigureInternalTokenizer wires up the in-process tokenization pool for
// examples that still drive the prompt-string indexer APIs.
//
// Deprecated: tokenize externally and call Indexer.ScoreTokens.
func ConfigureInternalTokenizer(config *kvcache.Config, modelName string) error {
	poolCfg, err := tokenization.DefaultConfig()
	if err != nil {
		return err
	}
	poolCfg.ModelName = modelName
	config.TokenizersPoolConfig = poolCfg
	ApplyTokenizerEndpoint(config)
	return nil
}

// ApplyTokenizerEndpoint reads TOKENIZER_ENDPOINT and overrides the UDS config
// on the given indexer config. No-op when TokenizersPoolConfig is nil.
func ApplyTokenizerEndpoint(config *kvcache.Config) {
	endpoint := os.Getenv(EnvTokenizerEndpoint)
	if endpoint == "" {
		return
	}
	if config.TokenizersPoolConfig == nil {
		return
	}
	config.TokenizersPoolConfig.UdsTokenizerConfig = &tokenization.UdsTokenizerConfig{
		SocketFile: endpoint,
		UseTCP:     isTCPAddr(endpoint),
	}
}

func getKVCacheIndexerConfig() (*kvcache.Config, error) {
	config, err := kvcache.NewDefaultConfig()
	if err != nil {
		return nil, err
	}

	if err := ConfigureInternalTokenizer(config, testdata.ModelName); err != nil {
		return nil, err
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
