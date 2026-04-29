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

package schedulerintegration

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
)

// TestSchedulerImportSurface_PreciseprefixcacheBuilds verifies the
// scheduler's preciseprefixcache scorer still builds and constructs
// against the in-tree kv-cache code. See issue #548.
func TestSchedulerImportSurface_PreciseprefixcacheBuilds(t *testing.T) {
	ctx := newTestContext(t)

	scorer := newScorer(ctx, t)
	var _ scheduling.Scorer = scorer

	processorCfg := kvblock.DefaultTokenProcessorConfig()
	require.NotNil(t, processorCfg)

	processor, err := kvblock.NewChunkedTokenDatabase(processorCfg)
	require.NoError(t, err)
	require.NotNil(t, processor)

	tokens := make([]uint32, processorCfg.BlockSize*3)
	for i := range tokens {
		tokens[i] = uint32(i) + 1 //nolint:gosec // bounded, fits uint32
	}
	keys, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, testModelName, nil)
	require.NoError(t, err)
	assert.NotEmpty(t, keys)
}
