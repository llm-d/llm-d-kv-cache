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
// current value onto each entry, Lookup filters entries whose stamp is below
// the current, Clear bumps. Stale entries are reclaimed lazily (LRU/cost
// pressure) or eagerly via Sweep.
type podGenTracker struct {
	gens        sync.Map      // map[string]*atomic.Uint64
	totalClears atomic.Uint64 // any pod ever cleared? — cheap fast-path predicate
}

func (g *podGenTracker) current(pod string) uint64 {
	if v, ok := g.gens.Load(pod); ok {
		return v.(*atomic.Uint64).Load()
	}
	return 0
}

func (g *podGenTracker) bump(pod string) uint64 {
	v, _ := g.gens.LoadOrStore(pod, new(atomic.Uint64))
	g.totalClears.Add(1)
	return v.(*atomic.Uint64).Add(1)
}

func (g *podGenTracker) anyClears() uint64 { return g.totalClears.Load() }

// genCache memoizes per-pod current generations within a single Lookup call,
// avoiding repeated sync.Map lookups when the same pod appears many times.
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
