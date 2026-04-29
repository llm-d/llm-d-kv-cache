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
	"context"
	"testing"

	"github.com/go-logr/logr/testr"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/scheduling/scorer/preciseprefixcache"
)

const testModelName = "test-model"

func newTestContext(t *testing.T) context.Context {
	t.Helper()
	return log.IntoContext(context.Background(), testr.New(t))
}

// newScorer constructs a preciseprefixcache.Scorer with default kv-cache
// and kvevents config. TokenizersPoolConfig is left nil and ZMQEndpoint
// empty, so no in-process tokenizer or live subscriber is started.
func newScorer(ctx context.Context, t *testing.T) *preciseprefixcache.Scorer {
	t.Helper()

	indexerConfig, err := kvcache.NewDefaultConfig()
	require.NoError(t, err)

	scorer, err := preciseprefixcache.New(ctx, preciseprefixcache.PluginConfig{
		IndexerConfig:  indexerConfig,
		KVEventsConfig: kvevents.DefaultConfig(),
	})
	require.NoError(t, err)
	require.NotNil(t, scorer)
	return scorer
}
