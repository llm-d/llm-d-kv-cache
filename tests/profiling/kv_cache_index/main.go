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
	"context"
	"flag"
	"fmt"
	"math/rand/v2"
	"time"

	"github.com/llm-d/llm-d-kv-cache-manager/pkg/kvcache/kvblock"
	"k8s.io/apimachinery/pkg/util/sets"
)

const modelName = "bert-base-uncased"

type IndexProfileResult struct {
	AddTime time.Duration
	LookupTime time.Duration
}

// TODO @kartikx: Use a more realistic workload if possible.
func generateWorkloadKeys(numKeys int) []kvblock.Key {
	// Uses time as seed to ensure that different profiling runs get different keys.
	randGen := rand.New(rand.NewPCG(42, uint64(time.Now().UnixNano())))

	keys := make([]kvblock.Key, numKeys)

	for i := range numKeys {
		keys[i] = kvblock.Key{
			ModelName: modelName,
			ChunkHash: randGen.Uint64(),
		}
	}

	return keys
}

func averageIndexProfileResults(durations []IndexProfileResult) IndexProfileResult {
	if len(durations) == 0 {
		return IndexProfileResult{}
	}

	var total IndexProfileResult
	for _, d := range durations {
		total.AddTime += d.AddTime
		total.LookupTime += d.LookupTime
	}

	count := time.Duration(len(durations))
	return IndexProfileResult{
		AddTime:    total.AddTime / count,
		LookupTime: total.LookupTime / count,
	}
}

func profileInMemoryIndex(numTrials, numKeys int) (IndexProfileResult, error) {
	return runProfileTrials(func() *kvblock.IndexConfig {
		return kvblock.DefaultIndexConfig()
	}, numTrials, numKeys)
}

func profileRedisIndex(numTrials, numKeys int) (IndexProfileResult, error) {
	return runProfileTrials(func() *kvblock.IndexConfig {
		return &kvblock.IndexConfig{
			RedisConfig:   kvblock.DefaultRedisIndexConfig(),
			EnableMetrics: false,
		}
	}, numTrials, numKeys)
}

func profileCostIndex(numTrials, numKeys int) (IndexProfileResult, error) {
	return runProfileTrials(func() *kvblock.IndexConfig {
		return &kvblock.IndexConfig{
			CostAwareMemoryConfig: kvblock.DefaultCostAwareMemoryIndexConfig(),
			EnableMetrics:         false,
		}
	}, numTrials, numKeys)
}

// runProfileTrials returns averaged results over multiple profiling runs.
func runProfileTrials(createConfig func() *kvblock.IndexConfig, numTrials, numKeys int) (IndexProfileResult, error) {
	profileResults := make([]IndexProfileResult, numTrials)

	for i := range numTrials {
		ctx := context.Background()

		indexConfig := createConfig()
		index, err := kvblock.NewIndex(ctx, indexConfig)
		if err != nil {
			return IndexProfileResult{}, fmt.Errorf("failed to create index: %w", err)
		}

		result, err := measureIndexRun(ctx, index, "pod1", numKeys)
		if err != nil {
			return IndexProfileResult{}, fmt.Errorf("failed to profile index: %w", err)
		}

		profileResults[i] = result
	}

	return averageIndexProfileResults(profileResults), nil
}

// measureIndexRun performs a single profiling measurement of Add and Lookup operations on an index.
func measureIndexRun(ctx context.Context, index kvblock.Index, podName string, numKeys int) (IndexProfileResult, error) {
	keys := generateWorkloadKeys(numKeys)

	podEntries := []kvblock.PodEntry{{PodIdentifier: podName, DeviceTier: "gpu"}}
	podIdentifierSet := sets.Set[string]{}

	addStartTime := time.Now()

	err := index.Add(ctx, keys, podEntries)
	if err != nil {
		return IndexProfileResult{}, fmt.Errorf("failed to add entries: %w", err)
	}

	addTime := time.Since(addStartTime)

	lookupStartTime := time.Now()

	_, err = index.Lookup(ctx, keys, podIdentifierSet)
	if err != nil {
		return IndexProfileResult{}, fmt.Errorf("failed to lookup entries: %w", err)
	}

	lookupTime := time.Since(lookupStartTime)

	return IndexProfileResult{
		AddTime: addTime,
		LookupTime: lookupTime,
	}, nil
}

func main() {
	var (
		numTrials = flag.Int("trials", 5, "Number of profiling trials to run")
		numKeys   = flag.Int("keys", 100, "Number of keys to use in each profiling run")
	)
	flag.Parse()

	result, err := profileCostIndex(*numTrials, *numKeys)
	if err != nil {
		fmt.Printf("Failed to profile cost index: %v\n", err)
	} else {
		fmt.Printf("[Cost Aware] Add: %v Lookup %v \n", result.AddTime, result.LookupTime)
	}

	result, err = profileRedisIndex(*numTrials, *numKeys)
	if err != nil {
		fmt.Printf("Failed to profile redis index: %v\n", err)
	} else {
		fmt.Printf("[Redis] Add: %v Lookup %v \n", result.AddTime, result.LookupTime)
	}

	result, err = profileInMemoryIndex(*numTrials, *numKeys)
	if err != nil {
		fmt.Printf("Failed to profile in memory index: %v\n", err)
	} else {
		fmt.Printf("[InMemory] Add: %v Lookup %v \n", result.AddTime, result.LookupTime)
	}
}
