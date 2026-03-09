//go:build !embedded_tokenizers

/*
Copyright 2026 The llm-d Authors.

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
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/docker/docker/api/types/container"
	"github.com/go-logr/logr/testr"
	"github.com/stretchr/testify/suite"
	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/wait"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	"github.com/llm-d/llm-d-kv-cache/pkg/utils"
)

const (
	defaultModelName = "ibm-granite/granite-3.1-8b-instruct"
	healthPort       = "8082/tcp"
	grpcPort         = "50051/tcp"
	//nolint:gosec // This is an environment variable name, not a credential
	envTokenizerImage = "UDS_TOKENIZER_IMAGE" // container image to use (set by Makefile)
)

// UDSTokenizerSuite defines a testify test suite for end-to-end testing
// of the KVCache indexer using a UDS-based tokenizer running in a container.
type UDSTokenizerSuite struct {
	suite.Suite

	// Suite-level: container and temp dir (shared across all tests)
	container   *testcontainers.DockerContainer
	grpcAddress string // host:port for gRPC connection

	config               *kvcache.Config
	tokenProcessorConfig *kvblock.TokenProcessorConfig
	tokenProcessor       kvblock.TokenProcessor
	kvBlockIndex         kvblock.Index
	indexer              *kvcache.Indexer

	tokenizer *tokenization.UdsTokenizer

	Pod1IP string
}

func (s *UDSTokenizerSuite) SetupSuite() {
	imageName := os.Getenv(envTokenizerImage)
	s.Require().NotEmpty(imageName, "%s must be set (run 'make e2e-test-uds')", envTokenizerImage)
	s.T().Logf("Using UDS tokenizer image: %s", imageName)

	s.container, s.grpcAddress = s.launchContainer(imageName)
	s.T().Logf("TCP tokenizer container started; gRPC at %s", s.grpcAddress)
}

//nolint:gocritic // nonamedreturns linter takes precedence
func (s *UDSTokenizerSuite) launchContainer(imageName string) (*testcontainers.DockerContainer, string) {
	ctx := context.Background()

	ctr, err := testcontainers.Run(ctx, imageName,
		testcontainers.WithExposedPorts(healthPort, grpcPort),
		testcontainers.WithEnv(map[string]string{
			"GRPC_PORT": "50051",
		}),
		testcontainers.WithHostConfigModifier(func(hc *container.HostConfig) {
			hc.AutoRemove = true
		}),
		testcontainers.WithWaitStrategyAndDeadline(120*time.Second,
			wait.ForHTTP("/health").WithPort(healthPort),
		),
	)
	s.Require().NoError(err, "failed to start UDS tokenizer container")

	mappedPort, err := ctr.MappedPort(ctx, "50051")
	s.Require().NoError(err, "failed to get mapped gRPC port")
	grpcAddress := fmt.Sprintf("localhost:%s", mappedPort.Port())

	return ctr, grpcAddress
}

// SetupTest initializes per-test components: context, tokenizer, indexer.
func (s *UDSTokenizerSuite) SetupTest() {
	testLogger := testr.New(s.T())
	log.SetLogger(testLogger)

	var err error
	s.config, err = kvcache.NewDefaultConfig()
	s.Require().NoError(err)

	// Configure UDS tokenizer to use TCP for testing
	s.config.TokenizersPoolConfig.ModelName = defaultModelName
	s.config.TokenizersPoolConfig.UdsTokenizerConfig = &tokenization.UdsTokenizerConfig{
		SocketFile: s.grpcAddress,
		UseTCP:     true,
	}

	s.tokenProcessorConfig = kvblock.DefaultTokenProcessorConfig()
	s.tokenProcessorConfig.BlockSize = 4
	s.tokenProcessor, err = kvblock.NewChunkedTokenDatabase(s.tokenProcessorConfig)
	s.Require().NoError(err)

	// Create the indexer - it will create its own UDS tokenizer pool with TCP
	s.indexer, err = kvcache.NewKVCacheIndexer(s.T().Context(), s.config, s.tokenProcessor)
	s.Require().NoError(err)
	s.kvBlockIndex = s.indexer.KVBlockIndex()

	// Also create a standalone tokenizer for direct testing
	udsTokenizer, err := tokenization.NewUdsTokenizer(
		s.T().Context(),
		&tokenization.UdsTokenizerConfig{
			SocketFile: s.grpcAddress,
			UseTCP:     true,
		},
		defaultModelName,
	)
	s.Require().NoError(err, "failed to create UDS tokenizer")
	s.tokenizer = udsTokenizer

	s.Pod1IP = "10.0.0.1"

	go s.indexer.Run(s.T().Context())
}

func (s *UDSTokenizerSuite) TearDownSuite() {
	if s.container != nil {
		if err := s.container.Terminate(context.Background()); err != nil {
			s.T().Logf("Warning: failed to terminate container: %v", err)
		}
	}
}

// promptToEngineAndRequestKeys tokenizes a prompt and returns its corresponding KV block keys.
//
//nolint:nonamedreturns // named returns for readability
func (s *UDSTokenizerSuite) promptToEngineAndRequestKeys(
	tokens []uint32,
) (engineKeys, requestKeys []kvblock.BlockHash) {
	requestKeys = s.tokenProcessor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, defaultModelName)
	s.Require().NotEmpty(requestKeys)

	engineKeys = s.tokenProcessor.TokensToKVBlockKeys(kvblock.BlockHash(1), tokens, defaultModelName)
	s.Require().NotEmpty(engineKeys)

	return engineKeys, requestKeys
}

// addEntriesToIndex adds block keys to the in-memory KV block index.
func (s *UDSTokenizerSuite) addEntriesToIndex(
	engineKeys, requestKeys []kvblock.BlockHash,
	podList []string,
) {
	s.Require().NotEmpty(engineKeys)
	s.Require().NotEmpty(requestKeys)

	err := s.kvBlockIndex.Add(s.T().Context(), engineKeys, requestKeys, utils.SliceMap(podList, func(pod string) kvblock.PodEntry {
		return kvblock.PodEntry{
			PodIdentifier:    pod,
			DeviceTier:       "gpu",
			DataParallelRank: kvblock.NoDataParallelRank,
		}
	}))
	s.Require().NoError(err)
}

// switchTokenizer creates a new UDS tokenizer for a different model and updates the suite.
func (s *UDSTokenizerSuite) switchTokenizer(modelName string) {
	udsTokenizer, err := tokenization.NewUdsTokenizer(
		s.T().Context(),
		&tokenization.UdsTokenizerConfig{
			SocketFile: s.grpcAddress,
			UseTCP:     true,
		},
		modelName,
	)
	s.Require().NoError(err, "failed to switch to model %s", modelName)

	s.tokenizer = udsTokenizer
	s.indexer.SetTokenizer(udsTokenizer, modelName)
}

// TestUDSTokenizerSuite runs the suite using testify's suite runner.
func TestUDSTokenizerSuite(t *testing.T) {
	suite.Run(t, new(UDSTokenizerSuite))
}
