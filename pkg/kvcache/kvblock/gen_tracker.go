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
	gens sync.Map // map[string]*atomic.Uint64
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
	return v.(*atomic.Uint64).Add(1)
}
