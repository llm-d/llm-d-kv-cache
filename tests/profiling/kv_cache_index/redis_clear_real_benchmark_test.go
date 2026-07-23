//go:build redis_real_bench

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

package main

import (
	"os"
	"testing"
)

const redisClearRealAddressEnv = "KV_CACHE_BENCH_REDIS_ADDR"

func BenchmarkRedisClearRealScaleLegacyFullScan(b *testing.B) {
	benchmarkRedisClearRealCases(b, false, redisClearScaleCases)
}

func BenchmarkRedisClearRealScaleReverseIndex(b *testing.B) {
	benchmarkRedisClearRealCases(b, true, redisClearScaleCases)
}

func BenchmarkRedisClearRealSharedKeyspaceLegacyFullScan(b *testing.B) {
	benchmarkRedisClearRealCases(b, false, redisClearSharedKeyspaceCases)
}

func BenchmarkRedisClearRealSharedKeyspaceReverseIndex(b *testing.B) {
	benchmarkRedisClearRealCases(b, true, redisClearSharedKeyspaceCases)
}

func BenchmarkRedisClearRealPodFanoutLegacyFullScan(b *testing.B) {
	benchmarkRedisClearRealCases(b, false, redisClearPodFanoutCases)
}

func BenchmarkRedisClearRealPodFanoutReverseIndex(b *testing.B) {
	benchmarkRedisClearRealCases(b, true, redisClearPodFanoutCases)
}

func benchmarkRedisClearRealCases(b *testing.B, reverseIndex bool, cases []redisClearBenchmarkCase) {
	b.Helper()
	address := os.Getenv(redisClearRealAddressEnv)
	if address == "" {
		b.Skipf("%s is required for real Redis benchmarks", redisClearRealAddressEnv)
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			benchmarkRedisClearWithAddress(b, address, tc, reverseIndex)
		})
	}
}
