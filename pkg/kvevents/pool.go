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
	"sync"

	"k8s.io/client-go/util/workqueue"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents/engineadapter"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents/events"
	"github.com/llm-d/llm-d-kv-cache/pkg/utils/logging"
)

const (
	defaultEventSourceDeviceTier = "GPU"
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
	queues         []workqueue.TypedRateLimitingInterface[*events.EventBatch]
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
		queues:         make([]workqueue.TypedRateLimitingInterface[*events.EventBatch], cfg.Concurrency),
		concurrency:    cfg.Concurrency,
		index:          index,
		tokenProcessor: tokenProcessor,
	}

	for i := 0; i < p.concurrency; i++ {
		p.queues[i] = workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[*events.EventBatch]())
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

// AddTask is called by the subscriber to add an event batch to the processing queue.
// It hashes the PodID to select a queue, ensuring events for the
// same pod always go to the same worker (ordered queue).
func (p *Pool) AddTask(batch *events.EventBatch) {
	// Use an FNV-1a hash to deterministically select a queue.
	// TODO: round-robin or simpler approach could be good enough
	h := fnv.New32a()
	_, err := h.Write([]byte(batch.Metadata.PodID))
	if err != nil {
		return
	}

	//nolint:gosec // if concurrency overflows then the world is in trouble anyway
	queueIndex := h.Sum32() % uint32(p.concurrency)
	p.queues[queueIndex].Add(batch)
}

// worker is the main processing loop for a single worker goroutine.
// It processes event batches from its dedicated queue using the workqueue pattern.
// TODO: profile and benchmark cases like backpressure, slow processing (profile), etc.
func (p *Pool) worker(ctx context.Context, workerIndex int) {
	defer p.wg.Done()
	queue := p.queues[workerIndex]
	for {
		batch, shutdown := queue.Get()
		if shutdown {
			return
		}

		// Use a nested func to ensure Done is always called.
		func(batch *events.EventBatch) {
			defer queue.Done(batch)
			p.processEventBatch(ctx, batch)
			// Task succeeded, remove it from the queue.
			queue.Forget(batch)
		}(batch)

		// Check if context was cancelled after processing a task.
		select {
		case <-ctx.Done():
			return
		default:
		}
	}
}

// processEventBatch processes a batch of generic events by calling each event's Process method.
func (p *Pool) processEventBatch(ctx context.Context, batch *events.EventBatch) {
	debugLogger := log.FromContext(ctx).V(logging.DEBUG)
	debugLogger.V(logging.TRACE).Info("Processing event batch",
		"podID", batch.Metadata.PodID,
		"modelName", batch.Metadata.ModelName,
		"engine", batch.Metadata.Engine,
		"eventCount", len(batch.Events))

	podIdentifier := batch.Metadata.PodID
	modelName := batch.Metadata.ModelName

	// Process each generic event in the batch
	for _, genericEvent := range batch.Events {
		if err := genericEvent.Process(ctx, p.index, p.tokenProcessor, podIdentifier, modelName); err != nil {
			debugLogger.Error(err, "Failed to process event",
				"eventType", genericEvent.Type(),
				"podIdentifier", podIdentifier,
				"modelName", modelName)
			// Continue processing other events even if one fails
		}
	}
}
