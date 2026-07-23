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

package kvblock_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
)

func TestCuckooStorageIndexCheckpointLifecycle(t *testing.T) {
	index := kvblock.NewCuckooStorageIndex(1_000)

	require.True(t, index.AddCheckpoint(123))
	assert.True(t, index.HasCheckpoint(123))
	require.True(t, index.AddCheckpoint(123), "duplicate insert should leave checkpoint present")

	index.RemoveCheckpoint(123)
	assert.False(t, index.HasCheckpoint(123))

	require.True(t, index.AddCheckpoint(456))
	index.Clear()
	assert.False(t, index.HasCheckpoint(456))
}

func TestCuckooStorageIndexStrideAndDefaultCapacity(t *testing.T) {
	index := kvblock.NewCuckooStorageIndex(0)

	assert.Equal(t, 0, index.Stride())
	index.SetStride(8)
	assert.Equal(t, 8, index.Stride())
	require.True(t, index.AddCheckpoint(789))
	assert.True(t, index.HasCheckpoint(789))
}
