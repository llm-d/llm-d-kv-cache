// Copyright 2025 The llm-d Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package kvcache //nolint:testpackage // Tests internal model registry implementation

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestModelRegistryDefaultBehavior verifies model registry default behavior.
// 1. If modelConfigs is not present (nil/empty) → all models are non-HMA
// 2. If modelConfigs is present but model not in list → non-HMA (default).
func TestModelRegistryDefaultBehavior(t *testing.T) {
	t.Run("NoConfig_AllModelsNonHMA", func(t *testing.T) {
		for _, configs := range [][]*ModelConfig{nil, {}} {
			registry := NewModelRegistry(configs)

			assert.False(t, registry.IsHMA("DeepSeek-V3"), "Unknown model should be non-HMA")
			assert.False(t, registry.IsHMA("Qwen/Qwen3-8B"), "Unknown model should be non-HMA")

			config := registry.GetModelConfig("unknown-model")
			require.NotNil(t, config)
			assert.Equal(t, "unknown-model", config.Name)
			assert.False(t, config.IsHMA)
			assert.Nil(t, config.AttentionGroups)
		}
	})

	t.Run("ModelNotInList_NonHMA", func(t *testing.T) {
		registry := NewModelRegistry([]*ModelConfig{
			{
				Name:  "DeepSeek-V3",
				IsHMA: true,
				AttentionGroups: []AttentionGroupConfig{
					{GroupID: 0, AttentionType: AttentionTypeFull, BlockSize: 64},
					{GroupID: 1, AttentionType: AttentionTypeSlidingWindow, BlockSize: 64, SlidingWindowSize: 4096},
				},
			},
		})

		assert.True(t, registry.IsHMA("DeepSeek-V3"), "Configured HMA model should be HMA")
		assert.False(t, registry.IsHMA("Qwen/Qwen3-8B"), "Model not in config should be non-HMA")
		assert.False(t, registry.IsHMA("unknown-model"), "Model not in config should be non-HMA")

		config := registry.GetModelConfig("Qwen/Qwen3-8B")
		require.NotNil(t, config)
		assert.Equal(t, "Qwen/Qwen3-8B", config.Name)
		assert.False(t, config.IsHMA)
	})

	t.Run("ModelInList_UseIsHMAFlag_False", func(t *testing.T) {
		registry := NewModelRegistry([]*ModelConfig{
			{
				Name:  "Qwen/Qwen3-8B",
				IsHMA: false,
			},
		})

		assert.False(t, registry.IsHMA("Qwen/Qwen3-8B"), "Should use IsHMA=false from config")

		config := registry.GetModelConfig("Qwen/Qwen3-8B")
		require.NotNil(t, config)
		assert.Equal(t, "Qwen/Qwen3-8B", config.Name)
		assert.False(t, config.IsHMA)
		assert.Nil(t, config.AttentionGroups)
	})

	t.Run("MixedConfig_CorrectBehavior", func(t *testing.T) {
		registry := NewModelRegistry([]*ModelConfig{
			{
				Name:  "DeepSeek-V3",
				IsHMA: true,
				AttentionGroups: []AttentionGroupConfig{
					{GroupID: 0, AttentionType: AttentionTypeFull, BlockSize: 64},
					{GroupID: 1, AttentionType: AttentionTypeSlidingWindow, BlockSize: 64, SlidingWindowSize: 4096},
				},
			},
			{
				Name:  "Qwen/Qwen3-8B",
				IsHMA: false,
			},
			{
				Name:  "meta-llama/Llama-3.1-8B",
				IsHMA: false,
			},
		})

		assert.True(t, registry.IsHMA("DeepSeek-V3"), "DeepSeek-V3 should be HMA")
		assert.False(t, registry.IsHMA("Qwen/Qwen3-8B"), "Qwen should be non-HMA")
		assert.False(t, registry.IsHMA("meta-llama/Llama-3.1-8B"), "Llama should be non-HMA")
		assert.False(t, registry.IsHMA("mistralai/Mistral-7B-v0.1"), "Unknown model should be non-HMA")
	})
}

func TestModelRegistryAttentionGroups(t *testing.T) {
	t.Run("NonHMAModel_NoAttentionGroups", func(t *testing.T) {
		registry := NewModelRegistry([]*ModelConfig{
			{Name: "Qwen/Qwen3-8B", IsHMA: false},
		})

		groups := registry.GetAttentionGroups("Qwen/Qwen3-8B")
		assert.Nil(t, groups, "Non-HMA model should have no attention groups")

		blockSize := registry.GetGroupBlockSize("Qwen/Qwen3-8B", 0)
		assert.Equal(t, 0, blockSize, "Non-HMA model should return 0 for block size")

		windowSize := registry.GetGroupSlidingWindow("Qwen/Qwen3-8B", 0)
		assert.Equal(t, 0, windowSize, "Non-HMA model should return 0 for window size")
	})

	t.Run("HMAModel_HasAttentionGroups", func(t *testing.T) {
		registry := NewModelRegistry([]*ModelConfig{
			{
				Name:  "DeepSeek-V3",
				IsHMA: true,
				AttentionGroups: []AttentionGroupConfig{
					{
						GroupID:       0,
						AttentionType: AttentionTypeFull,
						BlockSize:     64,
					},
					{
						GroupID:           1,
						AttentionType:     AttentionTypeSlidingWindow,
						BlockSize:         64,
						SlidingWindowSize: 4096,
					},
				},
			},
		})

		groups := registry.GetAttentionGroups("DeepSeek-V3")
		require.NotNil(t, groups)
		assert.Len(t, groups, 2)

		// Group 0: Full attention
		assert.Equal(t, 64, registry.GetGroupBlockSize("DeepSeek-V3", 0))
		assert.Equal(t, 0, registry.GetGroupSlidingWindow("DeepSeek-V3", 0))

		// Group 1: Sliding window
		assert.Equal(t, 64, registry.GetGroupBlockSize("DeepSeek-V3", 1))
		assert.Equal(t, 4096, registry.GetGroupSlidingWindow("DeepSeek-V3", 1))

		// Non-existent group
		assert.Equal(t, 0, registry.GetGroupBlockSize("DeepSeek-V3", 99))
	})

	t.Run("UnknownModel_NoAttentionGroups", func(t *testing.T) {
		registry := NewModelRegistry([]*ModelConfig{
			{Name: "DeepSeek-V3", IsHMA: true},
		})

		groups := registry.GetAttentionGroups("unknown-model")
		assert.Nil(t, groups, "Unknown model should have no attention groups")
	})
}

func TestModelRegistryRegisterModel(t *testing.T) {
	t.Run("RegisterNewModel", func(t *testing.T) {
		registry := NewModelRegistry(nil)

		assert.False(t, registry.IsHMA("new-model"))

		registry.RegisterModel(&ModelConfig{
			Name:  "new-model",
			IsHMA: true,
			AttentionGroups: []AttentionGroupConfig{
				{GroupID: 0, AttentionType: AttentionTypeFull, BlockSize: 32},
			},
		})

		assert.True(t, registry.IsHMA("new-model"))
		assert.Equal(t, 32, registry.GetGroupBlockSize("new-model", 0))
	})

	t.Run("UpdateExistingModel", func(t *testing.T) {
		registry := NewModelRegistry([]*ModelConfig{
			{Name: "test-model", IsHMA: false},
		})

		assert.False(t, registry.IsHMA("test-model"))

		registry.RegisterModel(&ModelConfig{
			Name:  "test-model",
			IsHMA: true,
			AttentionGroups: []AttentionGroupConfig{
				{GroupID: 0, AttentionType: AttentionTypeFull, BlockSize: 64},
			},
		})

		assert.True(t, registry.IsHMA("test-model"))
	})
}
