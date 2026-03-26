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

// online_uds demonstrates a KV-cache scoring service backed by a UDS tokenizer sidecar.
//
// It exposes two HTTP endpoints:
//
//	POST /score_completions      — score a /v1/completions request
//	POST /score_chat_completions — score a /v1/chat/completions request
//	GET  /metrics                — Prometheus metrics
//
// Environment variables:
//
//	MODEL_NAME      model to use (default: testdata.ModelName)
//	ZMQ_ENDPOINT    ZMQ endpoint for KV events (default: tcp://localhost:5557)
//	ZMQ_TOPIC       ZMQ topic filter (default: kv@)
//	POOL_CONCURRENCY worker count for the events pool (default: 4)
//	BLOCK_SIZE      token block size (default: from DefaultTokenProcessorConfig)
//	PYTHONHASHSEED  hash seed for token blocks (must match the vLLM sidecar)
//	HTTP_PORT       HTTP listen port (default: 8080)
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
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents/engineadapter"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"
)

const (
	envZMQEndpoint     = "ZMQ_ENDPOINT"
	envZMQTopic        = "ZMQ_TOPIC"
	envModelName       = "MODEL_NAME"
	envPoolConcurrency = "POOL_CONCURRENCY"
	envHTTPPort        = "HTTP_PORT"
	pythonHashSeed     = "PYTHONHASHSEED"
	blockSizeEnvVar    = "BLOCK_SIZE"

	defaultZMQEndpoint = "tcp://localhost:5557"
	defaultZMQTopic    = "kv@"
	defaultConcurrency = 4
	defaultHTTPPort    = "8080"
)

func main() {
	baseLogger := zap.New(zap.UseDevMode(true))
	log.SetLogger(baseLogger)

	ctx, cancel := context.WithCancel(log.IntoContext(context.Background(), baseLogger))
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigChan
		log.FromContext(ctx).Info("Received shutdown signal")
		cancel()
	}()

	if err := run(ctx); err != nil {
		log.FromContext(ctx).Error(err, "Failed to run KV-cache service")
	}
}

func run(ctx context.Context) error {
	logger := log.FromContext(ctx)

	kvCacheIndexer, err := setupKVCacheIndexer(ctx)
	if err != nil {
		return err
	}

	eventsPool := setupEventsPool(ctx, kvCacheIndexer.KVBlockIndex())
	eventsPool.Start(ctx)
	logger.Info("Events pool started")

	httpServer := startHTTPServer(ctx, kvCacheIndexer)
	logger.Info("HTTP server started", "addr", httpServer.Addr)

	<-ctx.Done()
	logger.Info("Shutting down...")

	// ctx is already cancelled here; shutdown requires a fresh context.
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()
	return httpServer.Shutdown(shutdownCtx) //nolint:contextcheck // ctx is already cancelled; shutdown needs a fresh context
}

func setupKVCacheIndexer(ctx context.Context) (*kvcache.Indexer, error) {
	cfg, err := kvcache.NewDefaultConfig()
	if err != nil {
		return nil, err
	}

	modelName := os.Getenv(envModelName)
	if modelName == "" {
		modelName = testdata.ModelName
	}
	cfg.TokenizersPoolConfig.ModelName = modelName
	cfg.KVBlockIndexConfig.EnableMetrics = true
	cfg.KVBlockIndexConfig.MetricsLoggingInterval = 30 * time.Second

	tokenProcessor, err := kvblock.NewChunkedTokenDatabase(tokenProcessorConfig())
	if err != nil {
		return nil, err
	}

	indexer, err := kvcache.NewKVCacheIndexer(ctx, cfg, tokenProcessor)
	if err != nil {
		return nil, err
	}

	go indexer.Run(ctx)
	log.FromContext(ctx).Info("KV cache indexer started", "model", modelName)
	return indexer, nil
}

func tokenProcessorConfig() *kvblock.TokenProcessorConfig {
	cfg := kvblock.DefaultTokenProcessorConfig()
	if seed := os.Getenv(pythonHashSeed); seed != "" {
		cfg.HashSeed = seed
	}
	if blockSize, err := strconv.Atoi(os.Getenv(blockSizeEnvVar)); err == nil && blockSize > 0 {
		cfg.BlockSize = blockSize
	}
	return cfg
}

func setupEventsPool(ctx context.Context, kvBlockIndex kvblock.Index) *kvevents.Pool {
	concurrency := defaultConcurrency
	if c, err := strconv.Atoi(os.Getenv(envPoolConcurrency)); err == nil && c > 0 {
		concurrency = c
	}
	zmqEndpoint := os.Getenv(envZMQEndpoint)
	if zmqEndpoint == "" {
		zmqEndpoint = defaultZMQEndpoint
	}
	zmqTopic := os.Getenv(envZMQTopic)
	if zmqTopic == "" {
		zmqTopic = defaultZMQTopic
	}

	tokenProcessor, err := kvblock.NewChunkedTokenDatabase(tokenProcessorConfig())
	if err != nil {
		log.FromContext(ctx).Error(err, "failed to create token processor for events pool")
		os.Exit(1)
	}
	return kvevents.NewPool(&kvevents.Config{
		Concurrency: concurrency,
		ZMQEndpoint: zmqEndpoint,
		TopicFilter: zmqTopic,
	}, kvBlockIndex, tokenProcessor, engineadapter.NewVLLMAdapter())
}

func startHTTPServer(ctx context.Context, indexer *kvcache.Indexer) *http.Server {
	logger := log.FromContext(ctx)
	mux := http.NewServeMux()

	mux.Handle("/metrics", promhttp.HandlerFor(ctrlmetrics.Registry, promhttp.HandlerOpts{EnableOpenMetrics: true}))

	mux.HandleFunc("/score_completions", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
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
		pods, err := indexer.GetPodScores(ctx, nil, req.Prompt, req.Model, nil)
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
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var req struct {
			Model    string               `json:"model"`
			Messages []types.Conversation `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid request body", http.StatusBadRequest)
			return
		}
		pods, err := indexer.GetPodScores(ctx, &types.RenderChatRequest{Conversation: req.Messages}, "", req.Model, nil)
		if err != nil {
			http.Error(w, fmt.Sprintf("error: %v", err), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(pods); err != nil {
			logger.Error(err, "failed to encode response")
		}
	})

	port := os.Getenv(envHTTPPort)
	if port == "" {
		port = defaultHTTPPort
	}
	server := &http.Server{
		Addr:              ":" + port,
		Handler:           mux,
		ReadHeaderTimeout: 20 * time.Second,
		ReadTimeout:       time.Minute,
		WriteTimeout:      time.Minute,
	}
	go func() {
		if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error(err, "HTTP server error")
		}
	}()
	return server
}
