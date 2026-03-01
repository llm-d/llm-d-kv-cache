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
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// AllBlocksClearedEvent represents all blocks being cleared from a pod's cache.
type AllBlocksClearedEvent struct {
	DeviceTier string
}

// Type returns the event type.
func (e *AllBlocksClearedEvent) Type() EventType {
	return EventTypeAllBlocksCleared
}

// Process processes the AllBlocksCleared event and updates the index.
// This removes all entries for the pod from the index.
func (e *AllBlocksClearedEvent) Process(ctx context.Context, index kvblock.Index,
	tokenProcessor kvblock.TokenProcessor, podIdentifier, modelName string) error {

	logger := log.FromContext(ctx)

	// For now, we just log the event.
	logger.Info("All blocks cleared event received",
		"podIdentifier", podIdentifier,
		"deviceTier", e.DeviceTier,
		"modelName", modelName)

	return nil
}
