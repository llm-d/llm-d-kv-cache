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
	"github.com/llm-d/llm-d-kv-cache/pkg/utils/logging"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// BlockRemovedEvent represents blocks being evicted from the cache.
// All hashes are already parsed to uint64 by the adapter.
type BlockRemovedEvent struct {
	BlockHashes []uint64
	DeviceTier  string
}

// Type returns the event type.
func (e *BlockRemovedEvent) Type() EventType {
	return EventTypeBlockRemoved
}

// Process processes the BlockRemoved event and updates the index.
func (e *BlockRemovedEvent) Process(ctx context.Context, index kvblock.Index,
	tokenProcessor kvblock.TokenProcessor, podIdentifier, modelName string) error {

	debugLogger := log.FromContext(ctx).V(logging.DEBUG)

	// Create PodEntry for this event's device tier
	podEntries := []kvblock.PodEntry{{
		PodIdentifier: podIdentifier,
		DeviceTier:    e.DeviceTier,
	}}

	// Evict each block
	for _, hash := range e.BlockHashes {
		engineKey := kvblock.BlockHash(hash)
		if err := index.Evict(ctx, engineKey, podEntries); err != nil {
			debugLogger.Error(err, "Failed to evict block from index",
				"engineKey", engineKey, "podIdentifier", podIdentifier)
			// Continue processing other blocks even if one fails
		}
	}

	return nil
}
