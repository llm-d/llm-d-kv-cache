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

package kvcache

import "sync"

type KVCacheBackendConfig struct {
	// Name is the identifier for this medium (e.g., "gpu", "cpu", "disk")
	Name string `json:"name"`
	// Weight is the scoring weight for blocks stored on this medium
	Weight float64 `json:"weight"`
}

func DefaultKVCacheBackendConfig() []*KVCacheBackendConfig {
	return []*KVCacheBackendConfig{
		{Name: "gpu", Weight: 1.0},
		{Name: "cpu", Weight: 0.8},
	}
}

// AttentionType defines the type of attention mechanism used by an attention group.
type AttentionType string

const (
	// AttentionTypeFull represents full/global attention (attends to all previous tokens).
	AttentionTypeFull AttentionType = "full"
	// AttentionTypeSlidingWindow represents sliding window attention (attends to last N tokens).
	AttentionTypeSlidingWindow AttentionType = "sliding_window"
	// AttentionTypeLocal represents local attention (attends to nearby tokens only).
	AttentionTypeLocal AttentionType = "local"
)

// AttentionGroupConfig holds configuration for a single attention group in HMA models.
type AttentionGroupConfig struct {
	// GroupID is the attention group identifier (e.g., 0 for full attention, 1 for sliding window)
	GroupID int `json:"groupId"`
	// AttentionType specifies the type of attention mechanism
	AttentionType AttentionType `json:"attentionType"`
	// BlockSize is the number of tokens per KV-cache block for this group
	BlockSize int `json:"blockSize"`
	// SlidingWindowSize is the window size for sliding window attention (0 or omitted for full attention)
	SlidingWindowSize int `json:"slidingWindowSize,omitempty"`
}

// ModelConfig holds the configuration for a specific model.
type ModelConfig struct {
	// Name is the model identifier (e.g., "Qwen/Qwen3-8B", "DeepSeek-V3")
	Name string `json:"name"`
	// IsHMA indicates whether this model uses Hybrid Multi-head Attention.
	// When true, StoredGroups tracking is enabled for cache entries.
	// When false, StoredGroups is left nil to save memory.
	IsHMA bool `json:"isHMA"`
	// AttentionGroups defines the attention group configuration for HMA models.
	// Only used when IsHMA is true.
	// Example for DeepSeek-V3:
	//   [{GroupID: 0, BlockSize: 64, SlidingWindowSize: 0},       // Full attention
	//    {GroupID: 1, BlockSize: 64, SlidingWindowSize: 4096}]    // Sliding window
	AttentionGroups []AttentionGroupConfig `json:"attentionGroups,omitempty"`
}

// ModelRegistry manages model configurations.
// It provides thread-safe access to model metadata needed for event processing.
type ModelRegistry struct {
	mu      sync.RWMutex
	configs map[string]*ModelConfig
}

// NewModelRegistry creates a new ModelRegistry with optional initial configs.
func NewModelRegistry(initialConfigs []*ModelConfig) *ModelRegistry {
	registry := &ModelRegistry{
		configs: make(map[string]*ModelConfig),
	}

	for _, config := range initialConfigs {
		registry.configs[config.Name] = config
	}

	return registry
}

// GetModelConfig retrieves the configuration for a given model name.
// If the model is not registered, it returns a default non-HMA config.
func (r *ModelRegistry) GetModelConfig(modelName string) *ModelConfig {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if config, exists := r.configs[modelName]; exists {
		return config
	}

	// Default: treat unknown models as non-HMA for memory efficiency
	return &ModelConfig{
		Name:  modelName,
		IsHMA: false,
	}
}

// RegisterModel adds or updates a model configuration.
func (r *ModelRegistry) RegisterModel(config *ModelConfig) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.configs[config.Name] = config
}

// IsHMA checks if a model uses Hybrid Multi-head Attention.
// Returns false for unknown models.
func (r *ModelRegistry) IsHMA(modelName string) bool {
	return r.GetModelConfig(modelName).IsHMA
}

// GetAttentionGroups returns the attention group configuration for a model.
// Returns nil for simple (non-HMA) models or unknown models.
func (r *ModelRegistry) GetAttentionGroups(modelName string) []AttentionGroupConfig {
	config := r.GetModelConfig(modelName)
	if !config.IsHMA {
		return nil
	}
	return config.AttentionGroups
}

// GetGroupBlockSize returns the block size for a specific attention group.
// Returns 0 if the model or group is not found.
func (r *ModelRegistry) GetGroupBlockSize(modelName string, groupID int) int {
	groups := r.GetAttentionGroups(modelName)
	for _, group := range groups {
		if group.GroupID == groupID {
			return group.BlockSize
		}
	}
	return 0
}

// GetGroupSlidingWindow returns the sliding window size for a specific attention group.
// Returns 0 for full attention groups or if not found.
func (r *ModelRegistry) GetGroupSlidingWindow(modelName string, groupID int) int {
	groups := r.GetAttentionGroups(modelName)
	for _, group := range groups {
		if group.GroupID == groupID {
			return group.SlidingWindowSize
		}
	}
	return 0
}

// NewDefaultModelRegistry creates a ModelRegistry with common defaults.
// By default, all models are treated as non-HMA for memory efficiency.
func NewDefaultModelRegistry() *ModelRegistry {
	return NewModelRegistry(nil)
}
