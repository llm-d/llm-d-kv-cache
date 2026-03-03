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

package kvevents

import (
	"context"
	"hash/fnv"
	"strings"
	"sync"

	"k8s.io/client-go/util/workqueue"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents/engineadapter"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents/events"
	"github.com/llm-d/llm-d-kv-cache/pkg/utils/logging"
)

const (
	defaultEventSourceDeviceTier = "gpu"
	defaultPodSelector           = "llm-d.ai/inferenceServing=true"
)

// Config holds the configuration for the event processing pool.
type Config struct {
	// ZMQEndpoint is the ZMQ address to connect to (e.g., "tcp://indexer:5557").
	ZMQEndpoint string `json:"zmqEndpoint,omitempty"`
	// TopicFilter is the ZMQ subscription filter (e.g., "kv@").
	TopicFilter string `json:"topicFilter"`
	// Concurrency is the number of parallel workers to run.
	Concurrency int `json:"concurrency"`
	// DiscoverPods enables the Kubernetes pod reconciler for automatic
	// per-pod subscriber management. When enabled, the reconciler watches
	// Kubernetes pods and creates/removes ZMQ subscribers dynamically.
	DiscoverPods bool `json:"discoverPods"`
	// PodDiscoveryConfig holds the configuration for pod discovery.
	// Only used when DiscoverPods is true.
	PodDiscoveryConfig *PodDiscoveryConfig `json:"podDiscoveryConfig,omitempty"`
}

// PodDiscoveryConfig holds configuration for the Kubernetes pod reconciler.
type PodDiscoveryConfig struct {
	// PodLabelSelector is a label selector string for filtering which pods to watch.
	// Example: "app=vllm" or "app=vllm,tier=gpu"
	PodLabelSelector string `json:"podLabelSelector"`
	// PodNamespace limits the reconciler to watch pods in a specific namespace.
	// If empty, watches all namespaces (requires appropriate RBAC).
	PodNamespace string `json:"podNamespace,omitempty"`
	// SocketPort is the port number where LLM pods expose their ZMQ socket.
	// The reconciler will connect to tcp://<PodIP>:<SocketPort>
	// Default: 5557
	SocketPort int `json:"socketPort"`
	// EngineType specifies which LLM engine type this reconciler manages.
	// This determines which adapter will be used for subscribers.
	// Default: "vllm"
	EngineType string `json:"engineType"`
}

// DefaultPodReconcilerConfig returns a default configuration for the pod reconciler.
// Defaults to vLLM engine type.
func DefaultPodReconcilerConfig() *PodDiscoveryConfig {
	return &PodDiscoveryConfig{
		PodLabelSelector: defaultPodSelector,
		SocketPort:       5557,
		EngineType:       string(engineadapter.EngineTypeVLLM),
	}
}

// DefaultConfig returns a default configuration for the event processing pool.
func DefaultConfig() *Config {
	return &Config{
		TopicFilter:        "kv@",
		Concurrency:        4,
		DiscoverPods:       true,
		PodDiscoveryConfig: DefaultPodReconcilerConfig(),
	}
}

// Pool is a sharded worker pool that processes event batches from engine adapters.
// It ensures that events for the same PodIdentifier are processed in order.
type Pool struct {
	queues         []workqueue.TypedRateLimitingInterface[*engineadapter.RawMessage]
	concurrency    int // can replace use with len(queues)
	index          kvblock.Index
	tokenProcessor kvblock.TokenProcessor
	wg             sync.WaitGroup
}

// NewPool creates a Pool with a sharded worker setup.
// Subscribers are managed by SubscriberManager which is controlled by the pod
// reconciler.
func NewPool(cfg *Config, index kvblock.Index, tokenProcessor kvblock.TokenProcessor) *Pool {
	if cfg == nil {
		cfg = DefaultConfig()
	}

	p := &Pool{
		queues:         make([]workqueue.TypedRateLimitingInterface[*engineadapter.RawMessage], cfg.Concurrency),
		concurrency:    cfg.Concurrency,
		index:          index,
		tokenProcessor: tokenProcessor,
	}

	for i := 0; i < p.concurrency; i++ {
		p.queues[i] = workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[*engineadapter.RawMessage]())
	}

	return p
}

// Start begins the worker pool.
// It is non-blocking.
func (p *Pool) Start(ctx context.Context) {
	logger := log.FromContext(ctx)
	logger.Info("Starting sharded event processing pool", "workers", p.concurrency)

	p.wg.Add(p.concurrency)
	for i := 0; i < p.concurrency; i++ {
		// Each worker is given its own dedicated queue shard.
		go p.worker(ctx, i)
	}
}

// Shutdown gracefully stops the pool and its global subscriber if present.
func (p *Pool) Shutdown(ctx context.Context) {
	logger := log.FromContext(ctx)
	logger.Info("Shutting down event processing pool...")

	for _, queue := range p.queues {
		queue.ShutDown()
	}

	p.wg.Wait()
	logger.Info("event processing pool shut down.")
}

// AddTask is called by the subscriber to add a raw message to the processing queue.
// It hashes the PodIdentifier to select a queue, ensuring messages for the
// same pod always go to the same worker (ordered queue).
func (p *Pool) AddTask(task *engineadapter.RawMessage) {
	// Use an FNV-1a hash to deterministically select a queue.
	// TODO: round-robin or simpler approach could be good enough
	h := fnv.New32a()
	_, err := h.Write([]byte(task.PodID))
	if err != nil {
		return
	}

	//nolint:gosec // if concurrency overflows then the world is in trouble anyway
	queueIndex := h.Sum32() % uint32(p.concurrency)
	p.queues[queueIndex].Add(task)
}

// worker is the main processing loop for a single worker goroutine.
// It processes raw messages from its dedicated queue using the workqueue pattern.
// TODO: profile and benchmark cases like backpressure, slow processing (profile), etc.
func (p *Pool) worker(ctx context.Context, workerIndex int) {
	defer p.wg.Done()
	queue := p.queues[workerIndex]
	for {
		task, shutdown := queue.Get()
		if shutdown {
			return
		}

		// Use a nested func to ensure Done is always called.
		func(task *engineadapter.RawMessage) {
			defer queue.Done(task)
			p.processRawMessage(ctx, task)
			// Task succeeded, remove it from the queue.
			queue.Forget(task)
		}(task)

		// Check if context was cancelled after processing a task.
		select {
		case <-ctx.Done():
			return
		default:
		}
	}
}

// processRawMessage decodes the raw message payload using the adapter and processes the resulting event batch.
func (p *Pool) processRawMessage(ctx context.Context, msg *engineadapter.RawMessage) {
	logger := log.FromContext(ctx)

	// Decode the raw payload into an event batch using the adapter
	batch, err := msg.Adapter.DecodeMessageToEventBatch(msg)
	if err != nil {
		logger.Error(err, "Failed to decode message to event batch",
			"podID", msg.PodID,
			"modelName", msg.ModelName)
		return
	}

	p.processEventBatch(ctx, batch, msg.PodID, msg.ModelName)
}

// processEventBatch processes a batch of events using type switches.
func (p *Pool) processEventBatch(ctx context.Context, batch *events.EventBatch, podIdentifier, modelName string) {
	debugLogger := log.FromContext(ctx).V(logging.DEBUG)
	debugLogger.V(logging.TRACE).Info("Processing event batch",
		"podID", podIdentifier,
		"modelName", modelName,
		"eventCount", len(batch.Events))

	// Process each event in the batch
	for _, genericEvent := range batch.Events {
		switch ev := genericEvent.(type) {
		case *events.BlockStoredEvent:
			// Default to gpu.
			// For non-gpu events, vLLM KV event has a non-empty DeviceTier field.
			deviceTier := defaultEventSourceDeviceTier
			if ev.DeviceTier != "" {
				deviceTier = strings.ToLower(ev.DeviceTier)
			}

			// Use LoRA name as model identifier if available, otherwise fall back to base model name.
			effectiveModelName := modelName
			if ev.LoraName != nil && *ev.LoraName != "" {
				effectiveModelName = *ev.LoraName
			}

			// Create PodEntry for this specific event's device tier
			podEntries := []kvblock.PodEntry{{PodIdentifier: podIdentifier, DeviceTier: deviceTier}}

			// Convert block hashes to BlockHash type (already uint64 from adapter)
			engineKeys := make([]kvblock.BlockHash, len(ev.BlockHashes))
			for i, hash := range ev.BlockHashes {
				engineKeys[i] = kvblock.BlockHash(hash)
			}

			// Get parent request key if parent hash exists
			parentRequestKey := kvblock.EmptyBlockHash
			if ev.ParentHash != 0 {
				parentEngineKey := kvblock.BlockHash(ev.ParentHash)
				key, err := p.index.GetRequestKey(ctx, parentEngineKey)
				if err != nil {
					debugLogger.Error(err, "Failed to get request key for parent block",
						"parentEngineKey", parentEngineKey, "effectiveModelName", effectiveModelName)
					continue
				}
				parentRequestKey = key
			}

			requestKeys := p.tokenProcessor.TokensToKVBlockKeys(parentRequestKey, ev.Tokens, effectiveModelName)

			// Only proceed if we have valid keys to add.
			if len(engineKeys) > 0 {
				if err := p.index.Add(ctx, engineKeys, requestKeys, podEntries); err != nil {
					debugLogger.Error(err, "Failed to add event to index",
						"podIdentifier", podIdentifier, "event", ev)
					continue // Continue processing other events even if one fails
				}
			}

		case *events.BlockRemovedEvent:
			// Default to gpu.
			// For non-gpu events, vLLM KV event has a non-empty DeviceTier field.
			deviceTier := defaultEventSourceDeviceTier
			if ev.DeviceTier != "" {
				deviceTier = strings.ToLower(ev.DeviceTier)
			}

			// Create PodEntry for this specific event's device tier
			podEntries := []kvblock.PodEntry{{PodIdentifier: podIdentifier, DeviceTier: deviceTier}}

			// Evict each block (hashes already uint64 from adapter)
			for _, hash := range ev.BlockHashes {
				engineKey := kvblock.BlockHash(hash)
				if err := p.index.Evict(ctx, engineKey, podEntries); err != nil {
					debugLogger.Error(err, "Failed to remove event from index",
						"podIdentifier", podIdentifier, "event", ev)
					continue // Continue processing other events even if one fails
				}
			}

		case *events.AllBlocksClearedEvent:
			// For now, just log it
			debugLogger.Info("All blocks cleared event received",
				"podIdentifier", podIdentifier,
				"deviceTier", ev.DeviceTier,
				"modelName", modelName)

		default:
			debugLogger.Info("Unknown event", "podIdentifier", podIdentifier, "event", genericEvent)
		}
	}
}
