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

package kvblock

import (
	"context"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/metrics"
)

type instrumentedIndex struct {
	next Index
	// backend is the label value identifying the wrapped store backend
	// (in_memory, redis, valkey, cost_aware_memory).
	backend string
}

// NewInstrumentedIndex wraps an Index and emits metrics for Add, Evict, and
// Lookup. It derives the backend label from the wrapped index so that
// admissions, evictions, hit rate, and entry count are partitioned by store.
func NewInstrumentedIndex(next Index) Index {
	return &instrumentedIndex{next: next, backend: backendName(next)}
}

// sizeReporter is implemented by index backends that can report their current
// key count cheaply (in O(1)). Redis/Valkey do not implement it, since an
// accurate count would require a DBSIZE round-trip that also over-counts
// engine-key mappings.
type sizeReporter interface {
	Size() int
}

// backendName derives the backend label for a raw index implementation.
// Redis and Valkey share the *RedisIndex type and are distinguished only by
// its BackendType field, so valkey is labeled separately from redis.
func backendName(idx Index) string {
	switch b := idx.(type) {
	case *InMemoryIndex:
		return "in_memory"
	case *RedisIndex:
		if b.BackendType == "valkey" {
			return "valkey"
		}
		return "redis"
	case *CostAwareMemoryIndex:
		return "cost_aware_memory"
	default:
		return "unknown"
	}
}

func (m *instrumentedIndex) Add(ctx context.Context, engineKeys, requestKeys []BlockHash, entries []PodEntry) error {
	err := m.next.Add(ctx, engineKeys, requestKeys, entries)
	metrics.Admissions.WithLabelValues(m.backend).Add(float64(len(requestKeys)))
	m.updateEntryCount()
	return err
}

func (m *instrumentedIndex) Evict(ctx context.Context, key BlockHash, keyType KeyType, entries []PodEntry) error {
	err := m.next.Evict(ctx, key, keyType, entries)
	metrics.Evictions.WithLabelValues(m.backend).Add(float64(len(entries)))
	m.updateEntryCount()
	return err
}

func (m *instrumentedIndex) Lookup(
	ctx context.Context,
	requestKeys []BlockHash,
	podIdentifierSet sets.Set[string],
) (map[BlockHash][]PodEntry, error) {
	timer := prometheus.NewTimer(metrics.LookupLatency)
	defer timer.ObserveDuration()

	metrics.LookupRequests.Inc()

	pods, err := m.next.Lookup(ctx, requestKeys, podIdentifierSet)
	if err != nil {
		return nil, err
	}

	// Record the hit metrics and then publish the current entry count and hit
	// rate together, so the gauge reflects the state after this lookup. Done
	// inline (rather than via a goroutine) to keep the gauges consistent with
	// the counters; the result is bounded by the prefix-chain early stop.
	recordLookupHit(pods)
	m.updateEntryCount()

	return pods, nil
}

func (m *instrumentedIndex) GetRequestKey(ctx context.Context, engineKey BlockHash) (BlockHash, error) {
	return m.next.GetRequestKey(ctx, engineKey)
}

func (m *instrumentedIndex) Clear(ctx context.Context, podIdentifier string) error {
	err := m.next.Clear(ctx, podIdentifier)
	m.updateEntryCount()
	return err
}

// updateEntryCount reports the current key count and cumulative hit rate for
// the backend. Backends that cannot report a cheap, accurate size do not
// implement sizeReporter, so entry_count is left unset for them.
func (m *instrumentedIndex) updateEntryCount() {
	r, ok := m.next.(sizeReporter)
	if !ok {
		return
	}
	metrics.Entries.WithLabelValues(m.backend).Set(float64(r.Size()))

	var metric dto.Metric
	hits := 0.0
	requests := 0.0
	if err := metrics.LookupHits.Write(&metric); err == nil {
		hits = metric.GetCounter().GetValue()
	}
	metric = dto.Metric{}
	if err := metrics.LookupRequests.Write(&metric); err == nil {
		requests = metric.GetCounter().GetValue()
	}
	if requests > 0 {
		metrics.HitRate.WithLabelValues(m.backend).Set(hits / requests)
	}
}

// recordLookupHit computes per-pod hit counts from a Lookup result and records
// the maximum pod hit count and lookup hits.
func recordLookupHit(keyToPods map[BlockHash][]PodEntry) {
	podCount := make(map[string]int)
	for _, pods := range keyToPods {
		for _, p := range pods {
			podCount[p.PodIdentifier]++
		}
	}

	maxHit := 0
	for _, count := range podCount {
		if count > maxHit {
			maxHit = count
		}
	}

	metrics.MaxPodHitCount.Add(float64(maxHit))
	metrics.LookupHits.Add(float64(maxHit))
}
