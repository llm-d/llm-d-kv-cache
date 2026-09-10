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

package metrics

import (
	"context"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/metrics"
)

// backendLabel is the label name used to partition index metrics by the
// underlying store backend (in_memory, redis, valkey, cost_aware_memory).
const backendLabel = "backend"

var (
	Admissions = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: "kvcache", Subsystem: "index", Name: "admissions_total",
		Help: "Total number of KV-block admissions",
	}, []string{backendLabel})
	Evictions = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: "kvcache", Subsystem: "index", Name: "evictions_total",
		Help: "Total number of KV-block evictions",
	}, []string{backendLabel})

	// LookupRequests counts how many Lookup() calls have been made.
	LookupRequests = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: "kvcache", Subsystem: "index", Name: "lookup_requests_total",
		Help: "Total number of lookup calls",
	})
	// MaxPodHitCount counts the maximum cache hits on a single pod on Lookup().
	MaxPodHitCount = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: "kvcache", Subsystem: "index", Name: "max_pod_hit_count_total",
		Help: "Maximum cache hits on a single pod on Lookup()",
	})
	// LookupHits counts how many keys were found in the cache on Lookup().
	LookupHits = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: "kvcache", Subsystem: "index", Name: "lookup_hits_total",
		Help: "Number of keys found in the cache on Lookup()",
	})
	// LookupLatency logs latency of lookup calls.
	LookupLatency = prometheus.NewHistogram(prometheus.HistogramOpts{
		Namespace: "kvcache", Subsystem: "index", Name: "lookup_latency_seconds",
		Help:    "Latency of Lookup calls in seconds",
		Buckets: prometheus.DefBuckets,
	})
	// HitRate is the cumulative block-level hit ratio (hits / requests).
	HitRate = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "kvcache", Subsystem: "index", Name: "hit_rate",
		Help: "Cumulative block-level hit ratio (hits / requests) since process start",
	}, []string{backendLabel})
	// Entries is the current number of keys held in the index backend.
	Entries = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "kvcache", Subsystem: "index", Name: "entries",
		Help: "Current number of keys held in the index",
	}, []string{backendLabel})

	RenderChatTemplateLatency = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "kvcache", Subsystem: "tokenization", Name: "render_chat_template_latency_seconds",
		Help:    "Latency of RenderChatTemplate calls in seconds",
		Buckets: prometheus.DefBuckets,
	}, []string{"tokenizer"})
	TokenizationLatency = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "kvcache", Subsystem: "tokenization", Name: "tokenization_latency_seconds",
		Help:    "Latency of Tokenization calls in seconds",
		Buckets: prometheus.DefBuckets,
	}, []string{"tokenizer"})
	TokenizedTokensCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "kvcache", Subsystem: "tokenization", Name: "tokenized_tokens_total",
			Help: "Number of tokens tokenized",
		}, []string{"tokenizer"})

	// DedupRemovedHashesSuppressed counts individual block hashes whose removal
	// was suppressed by the kvevents reference-count dedup filter because another
	// announcement still references the block. This counts block hashes, not
	// BlockRemoved events.
	DedupRemovedHashesSuppressed = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: "kvcache", Subsystem: "kvevents", Name: "dedup_removed_hashes_suppressed_total",
		Help: "Block hashes whose removal was suppressed by the KV-event dedup filter (block hashes, not BlockRemoved events)",
	})
	// DedupRemovedHashesForwarded counts individual block hashes forwarded to the
	// index for eviction after passing the kvevents dedup filter. This counts
	// block hashes, not BlockRemoved events.
	DedupRemovedHashesForwarded = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: "kvcache", Subsystem: "kvevents", Name: "dedup_removed_hashes_forwarded_total",
		Help: "Block hashes forwarded for eviction after the KV-event dedup filter (block hashes, not BlockRemoved events)",
	})
)

// Collectors returns a slice of all registered Prometheus collectors.
func Collectors() []prometheus.Collector {
	return []prometheus.Collector{
		Admissions, Evictions,
		LookupRequests, LookupHits, LookupLatency, MaxPodHitCount,
		HitRate, Entries,
		RenderChatTemplateLatency, TokenizationLatency, TokenizedTokensCount,
		DedupRemovedHashesSuppressed, DedupRemovedHashesForwarded,
	}
}

var registerMetricsOnce = sync.Once{}

// Register registers all metrics with K8s registry.
func Register() {
	registerMetricsOnce.Do(func() {
		metrics.Registry.MustRegister(Collectors()...)
	})
}

// StartMetricsLogging spawns a goroutine that logs current metric values every
// interval.
func StartMetricsLogging(ctx context.Context, interval time.Duration) {
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for range ticker.C {
			logMetrics(ctx)
		}
	}()
}

// collectCounterTotal sums the counter values across all label series of a
// (possibly labeled) counter collector. It is used by the metrics log beat to
// report a single total even when the underlying counter is a CounterVec.
func collectCounterTotal(c prometheus.Collector) float64 {
	ch := make(chan prometheus.Metric, 16)
	c.Collect(ch)
	close(ch)

	var total float64
	for m := range ch {
		var metric dto.Metric
		if err := m.Write(&metric); err != nil {
			continue
		}
		if metric.Counter != nil {
			total += metric.Counter.GetValue()
		}
	}
	return total
}

func logMetrics(ctx context.Context) {
	var m dto.Metric

	admissions := collectCounterTotal(Admissions)
	evictions := collectCounterTotal(Evictions)

	err := LookupRequests.Write(&m)
	if err != nil {
		return
	}
	lookups := m.GetCounter().GetValue()

	var hitsMetric dto.Metric
	err = LookupHits.Write(&hitsMetric)
	if err != nil {
		return
	}
	hits := hitsMetric.GetCounter().GetValue()

	var latencyMetric dto.Metric
	err = LookupLatency.Write(&latencyMetric)
	if err != nil {
		return
	}
	latencyCount := latencyMetric.GetHistogram().GetSampleCount()
	latencySum := latencyMetric.GetHistogram().GetSampleSum()

	latencyAvg := 0.0
	if latencyCount > 0 {
		latencyAvg = latencySum / float64(latencyCount)
	}

	var maxPodHitMetric dto.Metric
	err = MaxPodHitCount.Write(&maxPodHitMetric)
	if err != nil {
		return
	}
	maxPodHitCount := maxPodHitMetric.GetCounter().GetValue()

	log.FromContext(ctx).WithName("metrics").Info("metrics beat",
		"admissions", admissions,
		"evictions", evictions,
		"lookups", lookups,
		"hits", hits,
		"max_pod_hit_count", maxPodHitCount,
		"latency_count", latencyCount,
		"latency_sum", latencySum,
		"latency_avg", latencyAvg,
	)
}
