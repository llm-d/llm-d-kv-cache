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

package transport

import "context"

// Transport defines the interface for receiving raw bytes from different
// transport protocols (ZMQ, HTTP, gRPC, etc.).
type Transport interface {
	// Connect establishes a connection to a remote endpoint.
	// Used for per-pod subscriber mode where we connect to specific pods.
	Connect(ctx context.Context, endpoint string) error

	// Bind listens on a local endpoint for incoming connections.
	// Used for global subscriber mode where multiple pods publish to us.
	Bind(ctx context.Context, endpoint string) error

	// Subscribe sets the topic filter for receiving messages.
	// The filter format depends on the transport implementation.
	Subscribe(topicFilter string) error

	// Receive blocks until a message is received or context is canceled.
	// Returns raw message parts from the transport. For protocols that support
	// multi-part messages (like ZMQ), this returns multiple byte slices.
	// For single-part protocols (like HTTP), this returns a slice with one element.
	Receive(ctx context.Context) ([][]byte, error)

	// Close closes the transport connection and releases resources.
	Close() error
}
