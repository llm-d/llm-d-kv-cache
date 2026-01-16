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
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"strings"
	"sync"

	"github.com/vmihailenco/msgpack/v5"
	"k8s.io/client-go/util/workqueue"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
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
	// SocketPort is the port number where vLLM pods expose their ZMQ socket.
	// The reconciler will connect to tcp://<PodIP>:<SocketPort>
	// Default: 5557
	SocketPort int `json:"socketPort"`
}

// DefaultPodReconcilerConfig returns a default configuration for the pod reconciler.
func DefaultPodReconcilerConfig() *PodDiscoveryConfig {
	return &PodDiscoveryConfig{
		PodLabelSelector: defaultPodSelector,
		SocketPort:       5557,
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

// Message represents a message that is read from a ZMQ topic.
type Message struct {
	Topic   string
	Payload []byte
	// Sequence number of the message
	Seq uint64
	// PodIdentifier is the identifier of the pod that sent the event.
	// This will be extracted from the ZMQ topic.
	PodIdentifier string
	// ModelName is the name of the model that is associated with this event.
	ModelName string
}

// Pool is a sharded worker pool that processes events from ZMQ subscribers.
// It ensures that events for the same PodIdentifier are processed in order.
type Pool struct {
	queues         []workqueue.TypedRateLimitingInterface[*Message]
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
		queues:         make([]workqueue.TypedRateLimitingInterface[*Message], cfg.Concurrency),
		concurrency:    cfg.Concurrency,
		index:          index,
		tokenProcessor: tokenProcessor,
	}

	for i := 0; i < p.concurrency; i++ {
		p.queues[i] = workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[*Message]())
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

// AddTask is called by the subscriber to add a message to the processing queue.
// It hashes the PodIdentifier to select a queue, ensuring messages for the
// same pod always go to the same worker (ordered queue).
func (p *Pool) AddTask(task *Message) {
	// Use an FNV-1a hash to deterministically select a queue.
	// TODO: round-robin or simpler approach could be good enough
	h := fnv.New32a()
	_, err := h.Write([]byte(task.PodIdentifier))
	if err != nil {
		return
	}

	//nolint:gosec // if concurrency overflows then the world is in trouble anyway
	queueIndex := h.Sum32() % uint32(p.concurrency)
	p.queues[queueIndex].Add(task)
}

// worker is the main processing loop for a single worker goroutine.
// It processes messages from its dedicated queue using the workqueue pattern.
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
		func(task *Message) {
			defer queue.Done(task)
			p.processEvent(ctx, task)
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

// processEvent deserializes the message payload and calls the appropriate
// index method based on the event type. It returns an error to trigger retries.
func (p *Pool) processEvent(ctx context.Context, msg *Message) {
	debugLogger := log.FromContext(ctx).V(logging.DEBUG)
	debugLogger.V(logging.TRACE).Info("Processing event", "topic", msg.Topic, "seq", msg.Seq)

	var eventBatch EventBatch
	if err := msgpack.Unmarshal(msg.Payload, &eventBatch); err != nil {
		// This is likely a "poison pill" message that can't be unmarshalled.
		// We log the error but return nil to prevent it from being retried indefinitely.
		debugLogger.Error(err, "Failed to unmarshal event batch, dropping message")
		return
	}

	events := make([]event, 0, len(eventBatch.Events))
	for _, rawEvent := range eventBatch.Events {
		event, err := UnmarshalKVEvent(rawEvent)
		if err != nil {
			debugLogger.Error(err, "Failed to unmarshal event, skipping")
			continue
		}
		events = append(events, event)
	}

	podIdentifier := msg.PodIdentifier
	modelName := msg.ModelName
	p.digestEvents(ctx, podIdentifier, modelName, events)
}

func (p *Pool) digestEvents(ctx context.Context, podIdentifier, modelName string,
	events []event,
) {
	debugLogger := log.FromContext(ctx).V(logging.DEBUG)
	debugLogger.V(logging.TRACE).Info("Digesting events", "count", len(events))

	// Process each event in the batch
	for _, event := range events {
		switch ev := event.(type) {
		case BlockStored:
			// Default to gpu.
			// For non-gpu events, vLLM KV event has a non-empty Medium field.
			deviceTier := defaultEventSourceDeviceTier
			if ev.Medium != nil {
				deviceTier = strings.ToLower(*ev.Medium)
			}

			// Use LoRA name as model identifier if available, otherwise fall back to base model name.
			effectiveModelName := modelName
			if ev.LoraName != nil && *ev.LoraName != "" {
				effectiveModelName = *ev.LoraName
			}

			// Create PodEntry for this specific event's device tier
			podEntries := []kvblock.PodEntry{{PodIdentifier: podIdentifier, DeviceTier: deviceTier}}

			// Create a slice to hold the processed keys.
			engineKeys := make([]kvblock.BlockHash, 0, len(ev.BlockHashes))

			// Iterate over the hashes, convert each one to uint64, and create a key.
			for _, rawHash := range ev.BlockHashes {
				hash, err := getHashAsUint64(rawHash)
				if err != nil {
					debugLogger.Error(err, "Failed to convert block hash for BlockStored event", "rawHash", rawHash)
					continue
				}
				engineKeys = append(engineKeys, kvblock.BlockHash(hash))
			}

			var parentRequestKey kvblock.BlockHash
			if ev.ParentBlockHash != nil {
				hash, err := getHashAsUint64(ev.ParentBlockHash)
				if err != nil {
					debugLogger.Error(err, "Failed to convert parent block hash for BlockStored event",
						"rawHash", ev.ParentBlockHash)
					continue
				}

				parentEngineKey := kvblock.BlockHash(hash)

				key, err := p.index.GetRequestKey(ctx, parentEngineKey)
				if err != nil {
					debugLogger.Error(err, "Failed to get request key for parent block",
						"parentEngineKey", parentEngineKey, "effectiveModelName", effectiveModelName)
					continue
				}
				parentRequestKey = key
			}

			requestKeys := p.tokenProcessor.TokensToKVBlockKeys(parentRequestKey, ev.TokenIds, effectiveModelName)

			// Only proceed if we have valid keys to add.
			if len(engineKeys) > 0 {
				if err := p.index.Add(ctx, engineKeys, requestKeys, podEntries); err != nil {
					debugLogger.Error(err, "Failed to add event to index",
						"podIdentifier", podIdentifier, "event", ev)
					continue // Continue processing other events even if one fails
				}
			}

		case BlockRemoved:
			// Default to gpu.
			// For non-gpu events, vLLM KV event has a non-empty Medium field.
			deviceTier := defaultEventSourceDeviceTier
			if ev.Medium != nil {
				deviceTier = strings.ToLower(*ev.Medium)
			}

			// Create PodEntry for this specific event's device tier
			podEntries := []kvblock.PodEntry{{PodIdentifier: podIdentifier, DeviceTier: deviceTier}}

			// Iterate over the hashes, convert each one to uint64, and evict the key.
			for _, rawHash := range ev.BlockHashes {
				hash, err := getHashAsUint64(rawHash)
				if err != nil {
					debugLogger.Error(err, "Failed to convert block hash for BlockRemoved event", "rawHash", rawHash)
					continue
				}
				engineKey := kvblock.BlockHash(hash)
				if err := p.index.Evict(ctx, engineKey, podEntries); err != nil {
					debugLogger.Error(err, "Failed to remove event from index",
						"podIdentifier", podIdentifier, "event", ev)
					continue // Continue processing other events even if one fails
				}
			}
		case AllBlocksCleared:
			continue
		default:
			debugLogger.Info("Unknown event", "podIdentifier", podIdentifier, "event", ev)
		}
	}
}

// getHashAsUint64 converts a block hash from an `any` type to a uint64.
// It handles legacy uint64 hashes and new []byte hashes by taking the last 8 bytes
// and interpreting them as a big-endian integer, matching vLLM's compatibility logic.
func getHashAsUint64(hash any) (uint64, error) {
	switch val := hash.(type) {
	case uint64:
		// Hash is already in the target format.
		return val, nil
	case int64:
		// msgpack can decode small integers as int64.
		//nolint:gosec // int64 to uint64 conversion is safe here
		return uint64(val), nil
	case []byte:
		if len(val) == 0 {
			return 0, fmt.Errorf("hash byte slice is empty")
		}
		// If the slice is 8 bytes or longer, use the last 8 bytes.
		if len(val) >= 8 {
			return binary.BigEndian.Uint64(val[len(val)-8:]), nil
		}
		// If the slice is shorter than 8 bytes, pad it with leading zeros.
		padded := make([]byte, 8)
		copy(padded[8-len(val):], val)
		return binary.BigEndian.Uint64(padded), nil
	default:
		return 0, fmt.Errorf("unsupported hash type: %T", val)
	}
}
