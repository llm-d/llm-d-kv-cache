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

package transport

import (
	"context"
	"fmt"
	"time"

	zmq "github.com/pebbe/zmq4"
)

const (
	// pollTimeout is how often the poller should time out to check for context cancellation.
	pollTimeout = 250 * time.Millisecond
)

// ZMQTransport implements the Transport interface using ZeroMQ PUB/SUB pattern.
type ZMQTransport struct {
	socket *zmq.Socket
	poller *zmq.Poller
}

// NewZMQTransport creates a new ZMQ transport instance.
// The socket is not created here; it is created on the first Connect or Bind
// call, ensuring the socket is always created and used within the same goroutine.
func NewZMQTransport() (*ZMQTransport, error) {
	return &ZMQTransport{}, nil
}

// ensureSocket creates a fresh SUB socket only if the current one is nil
// (i.e. was previously closed). If the socket is still valid it is reused as-is,
// avoiding unnecessary socket recreation on the first Connect/Bind call.
func (z *ZMQTransport) ensureSocket() error {
	if z.socket != nil {
		// Socket is still open — nothing to do.
		return nil
	}
	socket, err := zmq.NewSocket(zmq.SUB)
	if err != nil {
		return fmt.Errorf("failed to create ZMQ SUB socket: %w", err)
	}
	z.socket = socket
	z.poller = zmq.NewPoller()
	return nil
}

// Connect establishes a connection to a remote ZMQ PUB endpoint.
// If the socket was previously closed, a new socket is created automatically.
func (z *ZMQTransport) Connect(ctx context.Context, endpoint string) error {
	if err := z.ensureSocket(); err != nil {
		return err
	}
	if err := z.socket.Connect(endpoint); err != nil {
		return fmt.Errorf("failed to connect to endpoint %s: %w", endpoint, err)
	}
	z.poller.Add(z.socket, zmq.POLLIN)
	return nil
}

// Bind listens on a local endpoint for incoming ZMQ PUB connections.
// If the socket was previously closed, a new socket is created automatically.
func (z *ZMQTransport) Bind(ctx context.Context, endpoint string) error {
	if err := z.ensureSocket(); err != nil {
		return err
	}
	if err := z.socket.Bind(endpoint); err != nil {
		return fmt.Errorf("failed to bind to endpoint %s: %w", endpoint, err)
	}
	z.poller.Add(z.socket, zmq.POLLIN)
	return nil
}

// Subscribe sets the topic filter for receiving messages.
func (z *ZMQTransport) Subscribe(topicFilter string) error {
	if err := z.socket.SetSubscribe(topicFilter); err != nil {
		return fmt.Errorf("failed to subscribe to topic filter %s: %w", topicFilter, err)
	}
	return nil
}

// Receive blocks until a message is received or context is canceled.
// Returns the raw multi-part ZMQ message as a slice of byte slices.
func (z *ZMQTransport) Receive(ctx context.Context) ([][]byte, error) {
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Poll with timeout to allow checking context cancellation
		polled, err := z.poller.Poll(pollTimeout)
		if err != nil {
			return nil, fmt.Errorf("failed to poll ZMQ socket: %w", err)
		}

		if len(polled) == 0 {
			// Timeout, continue to check context
			continue
		}

		parts, err := z.socket.RecvMessageBytes(0)
		if err != nil {
			return nil, fmt.Errorf("failed to receive ZMQ message: %w", err)
		}

		return parts, nil
	}
}

// Close closes the ZMQ socket and releases resources.
// Sets socket to nil so ensureSocket() knows to create a fresh one on next use.
func (z *ZMQTransport) Close() error {
	if z.socket != nil {
		err := z.socket.Close()
		z.socket = nil
		return err
	}
	return nil
}
