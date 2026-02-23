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

package engineadapter

import (
	"context"
	"fmt"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents/decoder"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents/events"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents/transport"
)

// EngineType represents the type of LLM engine.
type EngineType string

const (
	// EngineTypeVLLM represents the vLLM engine.
	EngineTypeVLLM EngineType = "vllm"
)

// NewAdapter creates a new engine adapter based on the engine type.
func NewAdapter(engineType EngineType) (EngineAdapter, error) {
	// It looks useless right now but we're preparing for future support of other engines ;)
	switch engineType {
	case EngineTypeVLLM:
		return NewVLLMAdapter()
	default:
		return nil, fmt.Errorf("unknown engine type: %s", engineType)
	}
}

// EngineAdapter defines the interface for engine-specific adapters.
// Each inference engine has its own adapter implementation that handles
// engine-specific operations.
type EngineAdapter interface {
	// Transport returns the transport layer for receiving messages.
	Transport() transport.Transport

	// Decoder returns the decoder for parsing message payloads.
	Decoder() decoder.Decoder

	// getHashAsUint64 converts engine-specific hash formats to uint64.
	getHashAsUint64(raw any) (uint64, error)

	// ReceiveAndDecode receives a message from the transport, parses it,
	// decodes the payload, and returns a batch of generic events.
	ReceiveAndDecode(ctx context.Context) (*events.EventBatch, error)

	// Connect establishes a connection to a remote endpoint.
	Connect(ctx context.Context, endpoint string) error

	// Bind listens on a local endpoint for incoming connections.
	Bind(ctx context.Context, endpoint string) error

	// SubscribeToTopic sets the topic filter for receiving messages.
	SubscribeToTopic(topicFilter string) error

	// Close closes the adapter and releases all resources.
	Close() error
}
