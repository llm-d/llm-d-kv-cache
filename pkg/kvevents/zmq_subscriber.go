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

	// Cached topic parsing results. Since a subscriber typically receives
	// messages with the same topic repeatedly, we cache the last parsed
	// result to avoid re-parsing on every message.
	lastTopic         string
	lastPodIdentifier string
	lastModelName     string
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

		if len(polled) == 0 {
			continue
		}

		parts, err := sub.RecvMessageBytes(0)
		if err != nil {
			debugLogger.Error(err, "Failed to receive message from zmq subscriber", "endpoint", z.endpoint)
			break // Exit on receive error to reconnect
		}
		if len(parts) != 3 {
			debugLogger.Error(nil, "Unexpected message format from zmq subscriber, expected 3 parts", "parts", len(parts))
			continue
		}

		topic := string(parts[0])
		seq := binary.BigEndian.Uint64(parts[1])
		payload := parts[2]

		// Extract pod identifier from topic, assuming "kv@<pod-id>@<model-name>" format.
		// Cache the result so repeated identical topics skip parsing.
		podIdentifier, modelName, ok := z.resolveTopicIdentifiers(topic)
		if !ok {
			debugLogger.Error(nil, "Failed to extract identifiers from topic, expected format kv@<pod-id>@<model-name>", "topic", topic)
			continue
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

// resolveTopicIdentifiers returns the pod identifier and model name for a topic,
// using cached values when the topic matches the previous call.
//
//nolint:nonamedreturns // gocritic requires named results.
func (z *zmqSubscriber) resolveTopicIdentifiers(topic string) (podID, model string, ok bool) {
	if topic == z.lastTopic {
		return z.lastPodIdentifier, z.lastModelName, true
	}
	podIdentifier, modelName, ok := parseTopic(topic)
	if !ok {
		return "", "", false
	}
	z.lastTopic = topic
	z.lastPodIdentifier = podIdentifier
	z.lastModelName = modelName
	return podIdentifier, modelName, true
}

// parseTopic extracts the pod identifier and model name from a ZMQ topic string.
// It expects the format "kv@<pod-id>@<model-name>".
//
//nolint:nonamedreturns // gocritic requires named results.
func parseTopic(topic string) (podID, model string, ok bool) {
	parts := strings.Split(topic, "@")
	if len(parts) != 3 {
		return "", "", false
	}
	return parts[1], parts[2], true
}
