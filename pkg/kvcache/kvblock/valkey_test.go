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

package kvblock_test

import (
	"testing"

	"github.com/alicebob/miniredis/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	. "github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
)

// createValkeyIndexForTesting creates a new Valkey index with a mock server for testing.
func createValkeyIndexForTesting(t *testing.T) Index {
	t.Helper()
	server, err := miniredis.Run()
	require.NoError(t, err)

	// Store server reference for cleanup
	t.Cleanup(func() {
		server.Close()
	})

	valkeyConfig := &RedisIndexConfig{
		Address:     server.Addr(),
		BackendType: "valkey",
		EnableRDMA:  false,
	}
	index, err := NewValkeyIndex(valkeyConfig)
	require.NoError(t, err)
	return index
}

// TestValkeyIndexBehavior tests the Valkey index implementation using common test behaviors.
func TestValkeyIndexBehavior(t *testing.T) {
	testCommonIndexBehavior(t, createValkeyIndexForTesting)
}

// TestValkeyIndexConfiguration tests Valkey-specific configuration options.
func TestValkeyIndexConfiguration(t *testing.T) {
	server, err := miniredis.Run()
	require.NoError(t, err)
	defer server.Close()

	tests := []struct {
		name            string
		config          *RedisIndexConfig
		expectedBackend string
		shouldSucceed   bool
	}{
		{
			name:            "default valkey config",
			config:          nil,
			expectedBackend: "valkey",
			shouldSucceed:   true,
		},
		{
			name: "valkey with explicit config",
			config: &RedisIndexConfig{
				Address:     server.Addr(),
				BackendType: "valkey",
				EnableRDMA:  false,
			},
			expectedBackend: "valkey",
			shouldSucceed:   true,
		},
		{
			name: "valkey with RDMA enabled",
			config: &RedisIndexConfig{
				Address:     server.Addr(),
				BackendType: "valkey",
				EnableRDMA:  true,
			},
			expectedBackend: "valkey",
			shouldSucceed:   true,
		},
		{
			name: "valkey:// URL scheme",
			config: &RedisIndexConfig{
				Address:     "valkey://" + server.Addr(),
				BackendType: "valkey",
				EnableRDMA:  false,
			},
			expectedBackend: "valkey",
			shouldSucceed:   true,
		},
		{
			name: "valkeys:// SSL URL scheme",
			config: &RedisIndexConfig{
				Address:     "valkeys://" + server.Addr(),
				BackendType: "valkey",
				EnableRDMA:  false,
			},
			expectedBackend: "valkey",
			shouldSucceed:   false, // miniredis doesn't support SSL
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var index Index
			var err error

			if tt.config == nil {
				// Test default config
				defaultConfig := DefaultValkeyIndexConfig()
				defaultConfig.Address = server.Addr() // Override for test server
				index, err = NewValkeyIndex(defaultConfig)
			} else {
				index, err = NewValkeyIndex(tt.config)
			}

			if tt.shouldSucceed {
				require.NoError(t, err)
				require.NotNil(t, index)

				// Verify the backend type is set correctly
				valkeyIndex, ok := index.(*RedisIndex)
				require.True(t, ok)
				assert.Equal(t, tt.expectedBackend, valkeyIndex.BackendType)

				if tt.config != nil && tt.config.EnableRDMA {
					assert.True(t, valkeyIndex.EnableRDMA)
				}
			} else {
				require.Error(t, err)
			}
		})
	}
}

// TestValkeyRedisCompatibility ensures that Valkey index behaves identically to Redis index.
func TestValkeyRedisCompatibility(t *testing.T) {
	server, err := miniredis.Run()
	require.NoError(t, err)
	defer func() {
		server.Close()
	}()

	// Create both Redis and Valkey indices with the same server
	redisConfig := &RedisIndexConfig{
		Address:     server.Addr(),
		BackendType: "redis",
	}
	redisIndex, err := NewRedisIndex(redisConfig)
	require.NoError(t, err)

	valkeyConfig := &RedisIndexConfig{
		Address:     server.Addr(),
		BackendType: "valkey",
	}
	valkeyIndex, err := NewValkeyIndex(valkeyConfig)
	require.NoError(t, err)

	// Both should be able to connect to the same mock server
	// and perform the same operations
	assert.IsType(t, &RedisIndex{}, redisIndex)
	assert.IsType(t, &RedisIndex{}, valkeyIndex)

	// Verify they have different backend types
	redisImpl, ok := redisIndex.(*RedisIndex)
	require.True(t, ok, "redisIndex should be of type *RedisIndex")
	valkeyImpl, ok := valkeyIndex.(*RedisIndex)
	require.True(t, ok, "valkeyIndex should be of type *RedisIndex")

	assert.Equal(t, "redis", redisImpl.BackendType)
	assert.Equal(t, "valkey", valkeyImpl.BackendType)
}

// TestValkeyURLSchemeHandling tests various URL scheme transformations.
func TestValkeyURLSchemeHandling(t *testing.T) {
	server, err := miniredis.Run()
	require.NoError(t, err)
	defer func() {
		server.Close()
	}()

	tests := []struct {
		name        string
		inputAddr   string
		expectError bool
	}{
		{
			name:        "plain address",
			inputAddr:   server.Addr(),
			expectError: false,
		},
		{
			name:        "redis:// scheme",
			inputAddr:   "redis://" + server.Addr(),
			expectError: false,
		},
		{
			name:        "valkey:// scheme",
			inputAddr:   "valkey://" + server.Addr(),
			expectError: false,
		},
		{
			name:        "rediss:// SSL scheme",
			inputAddr:   "rediss://" + server.Addr(),
			expectError: true, // miniredis doesn't support SSL
		},
		{
			name:        "valkeys:// SSL scheme",
			inputAddr:   "valkeys://" + server.Addr(),
			expectError: true, // miniredis doesn't support SSL
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := &RedisIndexConfig{
				Address:     tt.inputAddr,
				BackendType: "valkey",
			}

			index, err := NewValkeyIndex(config)

			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, index)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, index)
			}
		})
	}
}
