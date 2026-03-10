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

// AttentionGroupConfig defines the configuration for an attention group.
type AttentionGroupConfig struct {
	// WindowSize is the window size for SWA (Sliding Window Attention) groups.
	// nil means full-attention (no window limit).
	WindowSize *int `json:"windowSize"`
}

// ModelConfig defines model-specific KV cache configuration.
// Each model can have different attention architectures and block sizes.
type ModelConfig struct {
	// ModelName is the identifier for this model (e.g., "deepseek-r1", "llama-3")
	ModelName string `json:"modelName"`
	// BlockSize is the number of tokens per block (required for HMA models)
	BlockSize int `json:"blockSize"`
	// AttentionGroups maps group ID to attention configuration (for HMA models)
	// Group 0 is always full-attention (required for HMA)
	// Other groups are SWA with their respective window sizes
	// nil or empty map means standard non-HMA model (full-attention only)
	AttentionGroups map[int]*AttentionGroupConfig `json:"attentionGroups,omitempty"`
}

type KVCacheBackendConfig struct {
	// Name is the identifier for this medium (e.g., "gpu", "cpu", "disk")
	Name string `json:"name"`
	// Weight is the scoring weight for blocks stored on this medium
	Weight float64 `json:"weight"`
	// ModelConfigs maps model name to model-specific configuration
	// If empty, uses default configuration for all models
	ModelConfigs map[string]*ModelConfig `json:"modelConfigs,omitempty"`
}

func DefaultKVCacheBackendConfig() []*KVCacheBackendConfig {
	return []*KVCacheBackendConfig{
		{Name: "gpu", Weight: 1.0},
		{Name: "cpu", Weight: 0.8},
	}
}
