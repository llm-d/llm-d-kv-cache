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

package kvcache_test

import (
	"testing"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache"
	"github.com/stretchr/testify/assert"
)

func TestBuildAttentionInfoTable(t *testing.T) {
	tests := []struct {
		name         string
		modelConfig  *kvcache.ModelConfig
		expectNil    bool
		expectedInfo *kvcache.ModelAttentionInfo
	}{
		{
			name: "HMA_FullAndSWA",
			modelConfig: &kvcache.ModelConfig{
				Name:  "DeepSeek-V3",
				IsHMA: true,
				AttentionGroups: []kvcache.AttentionGroupConfig{
					{GroupID: 0, AttentionType: kvcache.AttentionTypeFull, BlockSize: 64},
					{GroupID: 1, AttentionType: kvcache.AttentionTypeSlidingWindow, BlockSize: 64, SlidingWindowSize: 4096},
				},
			},
			expectedInfo: &kvcache.ModelAttentionInfo{
				FullGroupID:     0,
				SWAGroupIDs:     []int{1},
				SWAWindowBlocks: []int{64},
			},
		},
		{
			name: "NonHMA_ReturnsNil",
			modelConfig: &kvcache.ModelConfig{
				Name:  "Qwen3-8B",
				IsHMA: false,
			},
			expectNil: true,
		},
		{
			name: "HMA_FullOnly_ReturnsNil",
			modelConfig: &kvcache.ModelConfig{
				Name:  "TestModel",
				IsHMA: true,
				AttentionGroups: []kvcache.AttentionGroupConfig{
					{GroupID: 0, AttentionType: kvcache.AttentionTypeFull, BlockSize: 64},
				},
			},
			expectNil: true,
		},
		{
			name: "HMA_SWAOnly_ReturnsNil",
			modelConfig: &kvcache.ModelConfig{
				Name:  "TestModel",
				IsHMA: true,
				AttentionGroups: []kvcache.AttentionGroupConfig{
					{GroupID: 1, AttentionType: kvcache.AttentionTypeSlidingWindow, BlockSize: 64, SlidingWindowSize: 129},
				},
			},
			expectNil: true,
		},
		{
			name: "HMA_MultipleSWAGroups",
			modelConfig: &kvcache.ModelConfig{
				Name:  "TestModel",
				IsHMA: true,
				AttentionGroups: []kvcache.AttentionGroupConfig{
					{GroupID: 0, AttentionType: kvcache.AttentionTypeFull, BlockSize: 64},
					{GroupID: 1, AttentionType: kvcache.AttentionTypeSlidingWindow, BlockSize: 64, SlidingWindowSize: 129},
					{GroupID: 2, AttentionType: kvcache.AttentionTypeSlidingWindow, BlockSize: 32, SlidingWindowSize: 97},
				},
			},
			expectedInfo: &kvcache.ModelAttentionInfo{
				FullGroupID:     0,
				SWAGroupIDs:     []int{1, 2},
				SWAWindowBlocks: []int{2, 3},
			},
		},
		{
			name: "HMA_ZeroBlockSize_Ignored",
			modelConfig: &kvcache.ModelConfig{
				Name:  "TestModel",
				IsHMA: true,
				AttentionGroups: []kvcache.AttentionGroupConfig{
					{GroupID: 0, AttentionType: kvcache.AttentionTypeFull, BlockSize: 64},
					{GroupID: 1, AttentionType: kvcache.AttentionTypeSlidingWindow, BlockSize: 0, SlidingWindowSize: 129},
				},
			},
			expectNil: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			infoMap := kvcache.BuildAttentionInfo([]*kvcache.ModelConfig{tt.modelConfig})
			info := infoMap[tt.modelConfig.Name]
			if tt.expectNil {
				assert.Nil(t, info)
			} else {
				assert.NotNil(t, info)
				assert.Equal(t, tt.expectedInfo.FullGroupID, info.FullGroupID)
				assert.Equal(t, tt.expectedInfo.SWAGroupIDs, info.SWAGroupIDs)
				assert.Equal(t, tt.expectedInfo.SWAWindowBlocks, info.SWAWindowBlocks)
			}
		})
	}
}