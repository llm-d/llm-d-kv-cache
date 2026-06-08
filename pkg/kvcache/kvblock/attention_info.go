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

package kvblock

import "sync"

// AttentionInfo holds per-model HMA scoring metadata: which group is full
// attention, which are sliding-window, and their window thresholds in blocks.
type AttentionInfo struct {
	FullGroupID     int
	SWAGroupIDs     []int
	SWAWindowBlocks []int
}

// AttentionInfoRegistry is a thread-safe registry of per-model AttentionInfo.
type AttentionInfoRegistry struct {
	mu    sync.RWMutex
	infos map[string]*AttentionInfo
}

// NewAttentionInfoRegistry creates a new, empty AttentionInfoRegistry.
func NewAttentionInfoRegistry() *AttentionInfoRegistry {
	return &AttentionInfoRegistry{
		infos: make(map[string]*AttentionInfo),
	}
}

// Set registers AttentionInfo for a model name.
func (r *AttentionInfoRegistry) Set(modelName string, info *AttentionInfo) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.infos[modelName] = info
}

// Get returns the AttentionInfo for a model name, or nil if not registered.
func (r *AttentionInfoRegistry) Get(modelName string) *AttentionInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.infos[modelName]
}
