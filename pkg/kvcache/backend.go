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
	AttentionTypeFull          AttentionType = "full"
	AttentionTypeSlidingWindow AttentionType = "sliding_window"
)

// AttentionGroupConfig holds configuration for a single attention group in HMA models.
type AttentionGroupConfig struct {
	GroupID           int           `json:"groupId"`
	AttentionType     AttentionType `json:"attentionType"`
	BlockSize         int           `json:"blockSize"`
	SlidingWindowSize int           `json:"slidingWindowSize,omitempty"`
}

// ModelConfig holds the configuration for a specific model.
type ModelConfig struct {
	Name            string                 `json:"name"`
	IsHMA           bool                   `json:"isHMA"`
	AttentionGroups []AttentionGroupConfig `json:"attentionGroups,omitempty"`
}

// DefaultModelConfigs returns the default model configurations.
func DefaultModelConfigs() []*ModelConfig {
	return []*ModelConfig{
		{
			Name:  "openai/gpt-oss-20b",
			IsHMA: true,
			AttentionGroups: []AttentionGroupConfig{
				{GroupID: 0, AttentionType: AttentionTypeSlidingWindow, BlockSize: 64, SlidingWindowSize: 128},
				{GroupID: 1, AttentionType: AttentionTypeFull, BlockSize: 64},
			},
		},
	}
}

// ModelAttentionInfo holds precomputed attention group metadata for scoring.
type ModelAttentionInfo struct {
	FullGroupID     int
	SWAGroupIDs     []int
	SWAWindowBlocks []int
}

func cdiv(a, b int) int {
	return (a + b - 1) / b
}

func buildAttentionInfo(config *ModelConfig) *ModelAttentionInfo {
	if !config.IsHMA {
		return nil
	}
	info := &ModelAttentionInfo{FullGroupID: -1}
	for _, group := range config.AttentionGroups {
		switch group.AttentionType {
		case AttentionTypeFull:
			info.FullGroupID = group.GroupID
		case AttentionTypeSlidingWindow:
			if group.BlockSize > 0 && group.SlidingWindowSize > 0 {
				info.SWAGroupIDs = append(info.SWAGroupIDs, group.GroupID)
				info.SWAWindowBlocks = append(info.SWAWindowBlocks, cdiv(group.SlidingWindowSize-1, group.BlockSize))
			}
		}
	}
	if info.FullGroupID < 0 || len(info.SWAWindowBlocks) == 0 {
		return nil
	}
	return info
}

// BuildHMAModels returns which models in configs use HMA.
func BuildHMAModels(configs []*ModelConfig) map[string]bool {
	result := make(map[string]bool)
	for _, c := range configs {
		if c.IsHMA {
			result[c.Name] = true
		}
	}
	return result
}

// BuildAttentionInfo returns precomputed attention info for all HMA models
// that have valid full + SWA group configuration.
func BuildAttentionInfo(configs []*ModelConfig) map[string]*ModelAttentionInfo {
	result := make(map[string]*ModelAttentionInfo)
	for _, c := range configs {
		if info := buildAttentionInfo(c); info != nil {
			result[c.Name] = info
		}
	}
	return result
}
