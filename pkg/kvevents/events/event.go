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

package events

import (
	"context"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
)

// EventType represents the type of KV-cache event.
type EventType string

// For logs
const (
	// EventTypeBlockStored indicates blocks were added to cache.
	EventTypeBlockStored EventType = "block_stored"
	// EventTypeBlockRemoved indicates blocks were evicted from cache.
	EventTypeBlockRemoved EventType = "block_removed"
	// EventTypeAllBlocksCleared indicates entire cache was cleared.
	EventTypeAllBlocksCleared EventType = "all_blocks_cleared"
)

// GenericEvent represents a KV-cache events containing already-parsed data.
type GenericEvent interface {
	// Type returns the event type.
	Type() EventType

	// Process processes the event and updates the index.
	Process(ctx context.Context, index kvblock.Index, tokenProcessor kvblock.TokenProcessor,
		podIdentifier, modelName string) error
}

// Metadata contains information about the source of an event batch.
type Metadata struct {
	// Topic is the original transport topic.
	Topic string
	// PodID identifies the pod that generated these events.
	PodID string
	// ModelName is the model associated with these events.
	ModelName string
	// Sequence is the message sequence number from the transport.
	Sequence uint64
	// Engine identifies which inference engine generated these events.
	Engine string
}

// EventBatch represents a batch of generic events from an inference engine.
// This is the primary data structure passed from adapters to the pool for processing.
type EventBatch struct {
	Metadata  Metadata
	Timestamp float64
	Events    []GenericEvent
}
