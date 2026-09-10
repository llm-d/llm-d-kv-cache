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
	"context"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/sets"

	. "github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/metrics"
)

func createInstrumentedIndexForTesting(t *testing.T) Index {
	t.Helper()
	cfg := DefaultInMemoryIndexConfig()
	cfg.PodCacheSize = 1000 // for testConcurrentOperations
	index, err := NewInMemoryIndex(cfg)
	require.NoError(t, err)
	instrumented := NewInstrumentedIndex(index)
	assert.NotNil(t, instrumented)
	return instrumented
}

func TestNewInstrumentedIndex(t *testing.T) {
	// Wrap with instrumentation
	instrumented := createInstrumentedIndexForTesting(t)
	// Verify it implements Index interface
	assert.Implements(t, (*Index)(nil), instrumented)
}

func TestInstrumentedIndexBehavior(t *testing.T) {
	testCommonIndexBehavior(t, createInstrumentedIndexForTesting)
}

// gaugeValue reads a gauge value via the exported collector, matching on the
// backend label. The Entries/HitRate gauges are keyed only by backend, so a
// given backend has a single series.
func gaugeValue(t *testing.T, c prometheus.Collector, backend string) float64 {
	t.Helper()
	ch := make(chan prometheus.Metric, 16)
	c.Collect(ch)
	close(ch)
	for m := range ch {
		var metric dto.Metric
		require.NoError(t, m.Write(&metric))
		if metric.Gauge == nil {
			continue
		}
		for _, l := range metric.Label {
			if l.GetName() == "backend" && l.GetValue() == backend {
				return metric.Gauge.GetValue()
			}
		}
	}
	t.Fatalf("gauge for backend %q not found", backend)
	return 0
}

// counterValue reads a single-series (unlabeled) counter value.
func counterValue(t *testing.T, c prometheus.Metric) float64 {
	t.Helper()
	var metric dto.Metric
	require.NoError(t, c.Write(&metric))
	return metric.GetCounter().GetValue()
}

// labeledCounterValue reads a counter series for one backend. A missing series
// has the same observable value as an uninitialized Prometheus counter: zero.
func labeledCounterValue(t *testing.T, c prometheus.Collector, backend string) float64 {
	t.Helper()
	ch := make(chan prometheus.Metric, 16)
	c.Collect(ch)
	close(ch)
	for m := range ch {
		var metric dto.Metric
		require.NoError(t, m.Write(&metric))
		if metric.Counter == nil {
			continue
		}
		for _, l := range metric.Label {
			if l.GetName() == "backend" && l.GetValue() == backend {
				return metric.Counter.GetValue()
			}
		}
	}
	return 0
}

func TestInstrumentedIndexHitRateAndEntries(t *testing.T) {
	index, err := NewInMemoryIndex(DefaultInMemoryIndexConfig())
	require.NoError(t, err)
	instrumented := NewInstrumentedIndex(index)

	ctx := context.Background()
	key := BlockHash(1)
	entries := []PodEntry{{PodIdentifier: "pod1"}}

	// After an Add, the entries gauge reflects the number of keys held.
	require.NoError(t, instrumented.Add(ctx, nil, []BlockHash{key}, entries))
	require.Equal(t, 1.0, gaugeValue(t, metrics.Entries, "in_memory"))

	// Perform a hit then a miss. LookupHits/LookupRequests are process-global
	// cumulative counters, so verify hit_rate equals the live hits/requests
	// ratio rather than assuming a zero baseline.
	_, err = instrumented.Lookup(ctx, []BlockHash{key}, sets.Set[string]{})
	require.NoError(t, err)
	_, err = instrumented.Lookup(ctx, []BlockHash{BlockHash(999)}, sets.Set[string]{})
	require.NoError(t, err)

	requests := counterValue(t, metrics.LookupRequests)
	hits := counterValue(t, metrics.LookupHits)
	require.Greater(t, requests, 0.0)
	require.InDelta(t, hits/requests, gaugeValue(t, metrics.HitRate, "in_memory"), 1e-9)
}

func TestInstrumentedIndexBackendLabel(t *testing.T) {
	// Use the cost-aware-memory backend so this exercises a distinct backend
	// label (and its Size()) from the in_memory tests above.
	index, err := NewCostAwareMemoryIndex(&CostAwareMemoryIndexConfig{Size: "1MiB"})
	require.NoError(t, err)
	instrumented := NewInstrumentedIndex(index)

	ctx := context.Background()
	key := BlockHash(7)
	entry := []PodEntry{{PodIdentifier: "pod1"}}
	admissionsBefore := labeledCounterValue(t, metrics.Admissions, "cost_aware_memory")
	evictionsBefore := labeledCounterValue(t, metrics.Evictions, "cost_aware_memory")
	inMemoryAdmissionsBefore := labeledCounterValue(t, metrics.Admissions, "in_memory")
	inMemoryEvictionsBefore := labeledCounterValue(t, metrics.Evictions, "in_memory")

	// Add flushes the Ristretto buffer internally, so Size() reflects the new key.
	require.NoError(t, instrumented.Add(ctx, nil, []BlockHash{key}, entry))
	require.Equal(t, 1.0, gaugeValue(t, metrics.Entries, "cost_aware_memory"))
	require.Equal(t, admissionsBefore+1, labeledCounterValue(t, metrics.Admissions, "cost_aware_memory"))
	require.Equal(t, inMemoryAdmissionsBefore, labeledCounterValue(t, metrics.Admissions, "in_memory"))

	// Eviction empties the index, bringing the entries gauge back to 0.
	require.NoError(t, instrumented.Evict(ctx, key, RequestKey, entry))
	require.Equal(t, 0.0, gaugeValue(t, metrics.Entries, "cost_aware_memory"))
	require.Equal(t, evictionsBefore+1, labeledCounterValue(t, metrics.Evictions, "cost_aware_memory"))
	require.Equal(t, inMemoryEvictionsBefore, labeledCounterValue(t, metrics.Evictions, "in_memory"))
}
