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
	"strings"
	"time"

	zmq "github.com/pebbe/zmq4"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-kv-cache/pkg/utils/logging"
)

const (
	// How long to wait before retrying to connect.
	retryInterval = 5 * time.Second
	// How often the poller should time out to check for context cancellation.
	pollTimeout = 250 * time.Millisecond
)

// zmqSubscriber connects to a ZMQ publisher and forwards messages to a pool.
type zmqSubscriber struct {
	pool        *Pool
	endpoint    string
	remote      bool
	topicFilter string
}

// newZMQSubscriber creates a new ZMQ subscriber.
func newZMQSubscriber(pool *Pool, endpoint, topicFilter string, remote bool) *zmqSubscriber {
	return &zmqSubscriber{
		pool:        pool,
		endpoint:    endpoint,
		remote:      remote,
		topicFilter: topicFilter,
	}
}

// Start connects to a ZMQ PUB socket as a SUB, receives messages,
// wraps them in Message structs, and pushes them into the pool.
// This loop will run until the provided context is canceled.
func (z *zmqSubscriber) Start(ctx context.Context) {
	logger := log.FromContext(ctx).WithName("zmq-subscriber")

	for {
		select {
		case <-ctx.Done():
			logger.Info("shutting down zmq-subscriber")
			return
		default:
			// We run the subscriber in a separate function to handle socket
			// setup/teardown and connection retries cleanly.
			z.runSubscriber(ctx)
			// wait before retrying, unless the context has been canceled.
			select {
			case <-time.After(retryInterval):
				logger.Info("retrying zmq-subscriber")
			case <-ctx.Done():
				logger.Info("shutting down zmq-subscriber")
				return
			}
		}
	}
}

// runSubscriber connects to the ZMQ PUB socket, subscribes to the topic filter,
// and listens for messages.
func (z *zmqSubscriber) runSubscriber(ctx context.Context) {
	logger := log.FromContext(ctx).WithName("zmq-subscriber")
	sub, err := zmq.NewSocket(zmq.SUB)
	if err != nil {
		logger.Error(err, "Failed to create subscriber socket")
		return
	}
	defer sub.Close()

	// Bind for local endpoints, connect for remote ones.
	if !z.remote {
		if err := sub.Bind(z.endpoint); err != nil {
			logger.Error(err, "Failed to bind subscriber socket", "endpoint", z.endpoint)
			return
		}
		logger.Info("Bound subscriber socket", "endpoint", z.endpoint)
	} else {
		if err := sub.Connect(z.endpoint); err != nil {
			logger.Error(err, "Failed to connect subscriber socket", "endpoint", z.endpoint)
			return
		}
		logger.Info("Connected subscriber socket", "endpoint", z.endpoint)
	}

	if err := sub.SetSubscribe(z.topicFilter); err != nil {
		logger.Error(err, "Failed to subscribe to topic filter", "topic", z.topicFilter)
		return
	}

	poller := zmq.NewPoller()
	poller.Add(sub, zmq.POLLIN)
	debugLogger := logger.V(logging.DEBUG)

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		polled, err := poller.Poll(pollTimeout)
		if err != nil {
			debugLogger.Error(err, "Failed to poll zmq subscriber", "endpoint", z.endpoint)
			break // Exit on poll error to reconnect
		}

		if len(polled) > 0 {
			parts, err := sub.RecvMessageBytes(0)
			if err != nil {
				debugLogger.Error(err, "Failed to receive message from zmq subscriber", "endpoint", z.endpoint)
				break // Exit on receive error to reconnect
			}
			if len(parts) != 3 {
				debugLogger.Error(err, "Failed to receive message from zmq subscriber", "endpoint", z.endpoint)
				continue
			}
			topic := string(parts[0])
			seqBytes := parts[1]
			payload := parts[2]

			seq := binary.BigEndian.Uint64(seqBytes)

			// Extract pod identifier from topic, assuming "kv@<pod-id>@<model-name>" format
			// TODO: optimize this to not occur for every message
			topicParts := strings.Split(topic, "@")
			var podIdentifier, modelName string
			if len(topicParts) == 3 {
				podIdentifier = topicParts[1]
				modelName = topicParts[2]
			} else {
				debugLogger.Error(nil, "Failed to extract identifiers from topic, expected format kv@<pod-id>@<model-name>", "topic", topic)
				continue // Useless if we can't extract pod identifier
			}

			debugLogger.V(logging.TRACE).Info("Received message from zmq subscriber",
				"topic", topic,
				"seq", seq,
				"podIdentifier", podIdentifier,
				"modelName", modelName,
				"payloadSize", len(payload))

			z.pool.AddTask(&Message{
				Topic:         topic,
				Payload:       payload,
				Seq:           seq,
				PodIdentifier: podIdentifier,
				ModelName:     modelName,
			})
		}
	}
}
