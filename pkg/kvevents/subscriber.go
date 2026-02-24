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
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents/engineadapter"
	"github.com/llm-d/llm-d-kv-cache/pkg/utils/logging"
)

const (
	// How long to wait before retrying to connect.
	retryInterval = 5 * time.Second
)

// subscriber connects to an engine via an adapter and forwards messages to a pool.
type subscriber struct {
	pool        *Pool
	adapter     engineadapter.EngineAdapter
	endpoint    string
	remote      bool
	topicFilter string
}

// newSubscriber creates a new generic subscriber.
func newSubscriber(pool *Pool, adapter engineadapter.EngineAdapter, endpoint, topicFilter string, remote bool) *subscriber {
	return &subscriber{
		pool:        pool,
		adapter:     adapter,
		endpoint:    endpoint,
		remote:      remote,
		topicFilter: topicFilter,
	}
}

// Start connects to an engine publisher, receives messages,
// wraps them in Message structs, and pushes them into the pool.
// This loop will run until the provided context is canceled.
func (s *subscriber) Start(ctx context.Context) {
	logger := log.FromContext(ctx).WithName("subscriber")

	for {
		select {
		case <-ctx.Done():
			logger.Info("shutting down subscriber")
			return
		default:
			// We run the subscriber in a separate function to handle
			// setup/teardown and connection retries cleanly.
			s.runSubscriber(ctx)
			// wait before retrying, unless the context has been canceled.
			select {
			case <-time.After(retryInterval):
				logger.Info("retrying subscriber")
			case <-ctx.Done():
				logger.Info("shutting down subscriber")
				return
			}
		}
	}
}

// runSubscriber connects to the engine, subscribes to the topic filter,
// and listens for messages.
func (s *subscriber) runSubscriber(ctx context.Context) {
	logger := log.FromContext(ctx).WithName("subscriber")
	debugLogger := logger.V(logging.DEBUG)

	// Connect or bind based on mode
	var err error
	if s.remote {
		err = s.adapter.Connect(ctx, s.endpoint)
		if err != nil {
			logger.Error(err, "Failed to connect to endpoint", "endpoint", s.endpoint)
			return
		}
		logger.Info("Connected to endpoint", "endpoint", s.endpoint)
	} else {
		err = s.adapter.Bind(ctx, s.endpoint)
		if err != nil {
			logger.Error(err, "Failed to bind to endpoint", "endpoint", s.endpoint)
			return
		}
		logger.Info("Bound to endpoint", "endpoint", s.endpoint)
	}

	// Ensure cleanup
	defer func() {
		if err := s.adapter.Close(); err != nil {
			logger.Error(err, "Failed to close adapter")
		}
	}()

	// Subscribe to topic filter
	if err := s.adapter.SubscribeToTopic(s.topicFilter); err != nil {
		logger.Error(err, "Failed to subscribe to topic filter", "topic", s.topicFilter)
		return
	}

	// Receive messages in a loop
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		// Receive and decode message from adapter
		eventBatch, err := s.adapter.ReceiveAndDecode(ctx)
		if err != nil {
			if ctx.Err() != nil {
				// Context was canceled, exit gracefully
				return
			}
			debugLogger.Error(err, "Failed to receive and decode message", "endpoint", s.endpoint)
			break // Exit on receive error to reconnect
		}

		debugLogger.V(logging.TRACE).Info("Received event batch",
			"topic", eventBatch.Metadata.Topic,
			"seq", eventBatch.Metadata.Sequence,
			"podIdentifier", eventBatch.Metadata.PodID,
			"modelName", eventBatch.Metadata.ModelName,
			"eventCount", len(eventBatch.Events))

		// Push event batch directly to pool
		s.pool.AddTask(eventBatch)
	}
}
