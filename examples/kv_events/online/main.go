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
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/llm-d/llm-d-kv-cache/examples/testdata"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
	preprocessing "github.com/llm-d/llm-d-kv-cache/pkg/preprocessing/chat_completions"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"
)

const (
	envHFToken     = "HF_TOKEN"
	envZMQEndpoint = "ZMQ_ENDPOINT"
	envZMQTopic    = "ZMQ_TOPIC"

	envPoolConcurrency = "POOL_CONCURRENCY"
	defaultZMQEndpoint = "tcp://localhost:5557"
	defaultZMQTopic    = "kv@"
	defaultConcurrency = 4

	pythonHashSeed  = "PYTHONHASHSEED"
	blockSizeEnvVar = "BLOCK_SIZE"

	envHTTPPort     = "HTTP_PORT"
	defaultHTTPPort = "8080"

	envExternalTokenization = "EXTERNAL_TOKENIZATION"
)

// ChatCompletionsRequest holds the fields needed for chat-completions rendering.
type ChatCompletionsRequest struct {
	Model string `json:"model"`
	*preprocessing.RenderChatRequest
}

func main() {
	baseLogger := zap.New(zap.UseDevMode(true))
	log.SetLogger(baseLogger)

	ctxBase := log.IntoContext(context.Background(), baseLogger)
	ctx, cancel := context.WithCancel(ctxBase)
	defer cancel()

	logger := log.FromContext(ctx)

	// Setup graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigChan
		logger.Info("Received shutdown signal")
		cancel()
	}()

	if err := run(ctx); err != nil {
		logger.Error(err, "Failed to run unified KV-cache service")
		return
	}
}

func run(ctx context.Context) error {
	logger := log.FromContext(ctx)

	// Setup Python path environment for chat completions
	logger.Info("Setting up Python path environment...")
	if err := setupPythonPath(ctx); err != nil {
		logger.Error(err, "Failed to setup Python path")
		return err
	}

	// Setup chat-templating processor
	logger.Info("Initializing chat-templating processor...")
	chatTemplatingProcessor, err := setupChatTemplatingProcessor()
	if err != nil {
		logger.Error(err, "Failed to setup chat-templating processor")
		return err
	}
	defer chatTemplatingProcessor.Finalize()
	logger.Info("Chat-templating processor initialized successfully")

	// Setup KV Cache Indexer
	kvCacheIndexer, err := setupKVCacheIndexer(ctx)
	if err != nil {
		logger.Error(err, "failed to setup KVCacheIndexer")
		return err
	}

	// Setup events pool
	eventsPool := setupEventsPool(ctx, kvCacheIndexer.KVBlockIndex())
	eventsPool.Start(ctx)
	logger.Info("Events pool started and listening for ZMQ messages")

	// Setup HTTP server
	httpServer := setupUnifiedHTTPEndpoints(ctx, kvCacheIndexer)

	logger.Info("=== Online KV Events Example Started ===")
	logger.Info("HTTP server running on http://localhost:8080")
	logger.Info("Available endpoints:")
	logger.Info("  - POST /score_completions - Score /v1/completions requests")
	logger.Info("  - POST /score_chat_completions - Score /v1/chat_completions requests")
	logger.Info("  - GET /metrics - Prometheus metrics endpoint")

	// Wait for shutdown
	<-ctx.Done()
	logger.Info("Shutting down KV-cache service...")

	// Graceful shutdown with timeout
	shutdownCtx, shutdownCancel := context.WithTimeout(ctx, 30*time.Second)
	defer shutdownCancel()

	if err := httpServer.Shutdown(shutdownCtx); err != nil {
		logger.Error(err, "HTTP server shutdown error")
	}

	return nil
}

func setupPythonPath(ctx context.Context) error {
	logger := log.FromContext(ctx)

	// Check if PYTHONPATH is already set
	pythonPath := os.Getenv("PYTHONPATH")
	if pythonPath == "" {
		err := fmt.Errorf("PYTHONPATH environment variable must be set to run this example")
		logger.Error(err, "PYTHONPATH not set")
		return err
	}

	logger.Info("PYTHONPATH is set", "path", pythonPath)
	return nil
}

func setupChatTemplatingProcessor() (*preprocessing.ChatTemplatingProcessor, error) {
	processor := preprocessing.NewChatTemplatingProcessor()
	if err := processor.Initialize(); err != nil {
		return nil, fmt.Errorf("failed to initialize chat-templating processor: %w", err)
	}
	return processor, nil
}

func getKVCacheIndexerConfig() (*kvcache.Config, error) {
	config, err := kvcache.NewDefaultConfig()
	if err != nil {
		return nil, err
	}

	huggingFaceToken := os.Getenv(envHFToken)
	if huggingFaceToken != "" {
		config.TokenizersPoolConfig.HFTokenizerConfig.HuggingFaceToken = huggingFaceToken
	}

	config.TokenizersPoolConfig.ModelName = testdata.ModelName

	useExternalTokenization, err := strconv.ParseBool(os.Getenv(envExternalTokenization))
	if err == nil && useExternalTokenization {
		config.TokenizersPoolConfig.UdsTokenizerConfig = &tokenization.UdsTokenizerConfig{
			SocketFile: "/tmp/tokenizer/tokenizer-uds.socket",
		}
		config.TokenizersPoolConfig.HFTokenizerConfig = nil
	}

	config.KVBlockIndexConfig.EnableMetrics = true
	config.KVBlockIndexConfig.MetricsLoggingInterval = 30 * time.Second

	return config, nil
}

func getTokenProcessorConfig() *kvblock.TokenProcessorConfig {
	config := kvblock.DefaultTokenProcessorConfig()
	hashSeed := os.Getenv(pythonHashSeed)
	if hashSeed != "" {
		config.HashSeed = hashSeed
	}

	blockSize, err := strconv.Atoi(os.Getenv(blockSizeEnvVar))
	if err == nil && blockSize >= 0 {
		config.BlockSize = blockSize
	}
	return config
}

func getEventsPoolConfig() *kvevents.Config {
	concurrency := defaultConcurrency
	if envConcurrency := os.Getenv(envPoolConcurrency); envConcurrency != "" {
		if c, err := strconv.Atoi(envConcurrency); err == nil && c > 0 {
			concurrency = c
		}
	}

	zmqEndpoint := os.Getenv(envZMQEndpoint)
	if zmqEndpoint == "" {
		zmqEndpoint = defaultZMQEndpoint
	}

	zmqTopic := os.Getenv(envZMQTopic)
	if zmqTopic == "" {
		zmqTopic = defaultZMQTopic
	}

	return &kvevents.Config{
		Concurrency: concurrency,
		ZMQEndpoint: zmqEndpoint,
		TopicFilter: zmqTopic,
	}
}

func setupKVCacheIndexer(ctx context.Context) (*kvcache.Indexer, error) {
	logger := log.FromContext(ctx)

	cfg, err := getKVCacheIndexerConfig()
	if err != nil {
		return nil, err
	}

	kvCacheIndexer, err := kvcache.NewKVCacheIndexer(ctx, cfg,
		kvblock.NewChunkedTokenDatabase(getTokenProcessorConfig()))
	if err != nil {
		return nil, err
	}

	logger.Info("Created Indexer")

	go kvCacheIndexer.Run(ctx)
	logger.Info("Started Indexer")

	return kvCacheIndexer, nil
}

func setupEventsPool(ctx context.Context, kvBlockIndex kvblock.Index) *kvevents.Pool {
	logger := log.FromContext(ctx)

	cfg := getEventsPoolConfig()

	logger.Info("Creating events pool", "config", cfg)
	tokenProcessor := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
	pool := kvevents.NewPool(cfg, kvBlockIndex, tokenProcessor)

	return pool
}

func setupUnifiedHTTPEndpoints(
	ctx context.Context,
	kvCacheIndexer *kvcache.Indexer,
) *http.Server {
	logger := log.FromContext(ctx)

	mux := http.NewServeMux()

	mux.Handle("/metrics", promhttp.HandlerFor(ctrlmetrics.Registry, promhttp.HandlerOpts{
		EnableOpenMetrics: true,
	}))

	mux.HandleFunc("/score_completions", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Prompt string `json:"prompt"`
			Model  string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid JSON body", http.StatusBadRequest)
			return
		}
		if req.Prompt == "" {
			http.Error(w, "field 'prompt' required", http.StatusBadRequest)
			return
		}

		pods, err := kvCacheIndexer.GetPodScores(ctx, nil, req.Prompt, req.Model, nil)
		if err != nil {
			http.Error(w, fmt.Sprintf("error: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(pods); err != nil {
			logger.Error(err, "failed to encode response")
		}
	})

	mux.HandleFunc("/score_chat_completions", func(w http.ResponseWriter, r *http.Request) {
		logger.Info("Received request for /score_chat_completions", "body", r.Body)

		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req ChatCompletionsRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}

		pods, err := kvCacheIndexer.GetPodScores(ctx, req.RenderChatRequest, "", req.Model, nil)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to get score request: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(pods); err != nil {
			logger.Error(err, "Failed to encode score response")
			http.Error(w, "Internal server error", http.StatusInternalServerError)
			return
		}
	})

	// Get HTTP port
	httpPort := os.Getenv(envHTTPPort)
	if httpPort == "" {
		httpPort = defaultHTTPPort
	}

	server := &http.Server{
		Addr:              ":" + httpPort,
		Handler:           mux,
		ReadHeaderTimeout: 20 * time.Second,
		ReadTimeout:       1 * time.Minute,
		WriteTimeout:      1 * time.Minute,
	}

	// Start HTTP server in goroutine
	go func() {
		if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error(err, "HTTP server error")
		}
	}()

	return server
}
