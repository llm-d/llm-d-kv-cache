/*
Copyright 2025 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package kvblock

import (
	"sync"
	"sync/atomic"
)

// podGenTracker holds a per-pod monotonic generation counter. Add stamps the
// current generation onto each entry; Lookup filters entries whose stamped
// generation is below the current. Clear bumps the pod's generation, which
// invalidates all prior entries lazily.
//
// This is the in-process variant used by InMemoryIndex and CostAwareMemoryIndex.
type podGenTracker struct {
	gens         sync.Map      // map[string]*atomic.Uint64
	totalClears  atomic.Uint64 // monotonically incremented on every bump; cheap "anyone cleared?" predicate
}

// current returns the current generation for the pod (0 if pod is unknown).
func (g *podGenTracker) current(pod string) uint64 {
	if v, ok := g.gens.Load(pod); ok {
		return v.(*atomic.Uint64).Load()
	}
	return 0
}

// bump increments and returns the new generation for the pod.
func (g *podGenTracker) bump(pod string) uint64 {
	v, _ := g.gens.LoadOrStore(pod, new(atomic.Uint64))
	g.totalClears.Add(1)
	return v.(*atomic.Uint64).Add(1)
}

// anyClears returns the total number of Clear operations performed across all pods.
// Lookup uses this as a cheap fast-path predicate: if zero, no entry can be stale
// and per-entry generation filtering can be skipped.
func (g *podGenTracker) anyClears() uint64 {
	return g.totalClears.Load()
}

// genCache is a one-call-scoped memoization of per-pod current generations.
// Lookup builds one of these at call entry and reuses it across all entries
// in the call to avoid repeated sync.Map lookups in the hot path.
type genCache struct {
	tracker *podGenTracker
	cache   map[string]uint64
}

func (g *podGenTracker) snapshot() *genCache {
	return &genCache{tracker: g, cache: make(map[string]uint64)}
}

func (c *genCache) current(pod string) uint64 {
	if v, ok := c.cache[pod]; ok {
		return v
	}
	v := c.tracker.current(pod)
	c.cache[pod] = v
	return v
}
