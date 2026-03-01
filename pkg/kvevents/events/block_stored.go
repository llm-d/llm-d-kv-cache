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

// BlockStoredEvent represents blocks being added to the cache.
type BlockStoredEvent struct {
	BlockHashes []uint64
	Tokens      []uint32
	ParentHash  uint64
	DeviceTier  string
	LoraID      *int
	LoraName    *string
}

// Type returns the event type.
func (e *BlockStoredEvent) Type() EventType {
	return EventTypeBlockStored
}

// Process processes the BlockStored event and updates the index.
func (e *BlockStoredEvent) Process(ctx context.Context, index kvblock.Index,
	tokenProcessor kvblock.TokenProcessor, podIdentifier, modelName string) error {

	debugLogger := log.FromContext(ctx).V(logging.DEBUG)

	// Use LoRA name as model identifier if available, otherwise fall back to base model name
	effectiveModelName := modelName
	if e.LoraName != nil && *e.LoraName != "" {
		effectiveModelName = *e.LoraName
	}

	// Create PodEntry for this event's device tier
	podEntries := []kvblock.PodEntry{{
		PodIdentifier: podIdentifier,
		DeviceTier:    e.DeviceTier,
	}}

	// Convert block hashes to BlockHash type
	engineKeys := make([]kvblock.BlockHash, len(e.BlockHashes))
	for i, hash := range e.BlockHashes {
		engineKeys[i] = kvblock.BlockHash(hash)
	}

	// Get parent request key if parent hash exists
	parentRequestKey := kvblock.EmptyBlockHash
	if e.ParentHash != 0 {
		parentEngineKey := kvblock.BlockHash(e.ParentHash)
		key, err := index.GetRequestKey(ctx, parentEngineKey)
		if err != nil {
			debugLogger.Error(err, "Failed to get request key for parent block",
				"parentEngineKey", parentEngineKey)
		} else {
			parentRequestKey = key
		}
	}

	// Compute request keys from tokens using effective model name
	requestKeys := tokenProcessor.TokensToKVBlockKeys(parentRequestKey, e.Tokens, effectiveModelName)

	// Only proceed if we have valid keys to add.
	if len(engineKeys) > 0 {
		if err := index.Add(ctx, engineKeys, requestKeys, podEntries); err != nil {
			debugLogger.Error(err, "Failed to add blocks to index",
				"podIdentifier", podIdentifier, "modelName", modelName)
			return err
		}
	}

	return nil
}
