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

package engineadapter

import (
	"context"
	"fmt"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents/events"
)

// EngineType represents the type of LLM engine.
type EngineType string

const (
	// EngineTypeVLLM represents the vLLM engine.
	EngineTypeVLLM EngineType = "vllm"
)

// RawMessage holds the pre-parsed framing metadata from a received transport
// message, with the payload still in raw (not yet decoded) bytes.
// It is returned by ReceiveMessage and passed to DecodeMessageToEventBatch.
type RawMessage struct {
	// PodID that is parsed from the topic.
	PodID string
	// Model name that is parsed from the topic.
	ModelName string
	// Sequence is the message sequence number from the transport.
	Sequence uint64
	// Topic is the original transport topic string.
	Topic string
	// Payload is the raw msgpack-encoded event batch bytes, not yet decoded.
	Payload []byte
	// Adapter is the engine adapter that can decode this payload.
	Adapter EngineAdapter
}

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
// engine-specific message receiving, decoding, and connection management.
type EngineAdapter interface {
	// ReceiveMessage receives a raw message and returns a RawMessage
	// with pre-parsed framing metadata, but with the payload still in raw bytes.
	// This is intentionally cheap — no event payload decoding happens here.
	ReceiveMessage(ctx context.Context) (*RawMessage, error)

	// DecodeMessageToEventBatch decodes the raw payload of a RawMessage into a
	// fully populated EventBatch.
	DecodeMessageToEventBatch(msg *RawMessage) (*events.EventBatch, error)

	// Setup establishes a connection (or binding) to an endpoint and subscribes to a topic.
	// If remote is true, it connects to a remote endpoint; otherwise, it binds locally.
	Setup(ctx context.Context, endpoint, topicFilter string, remote bool) error

	// Close closes the adapter and releases all resources.
	Close() error
}
