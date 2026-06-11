/*
Copyright 2026 The llm-d Authors.

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

package kvblock

import "sync"

// GroupID identifies a vLLM KV cache group.
type GroupID int

// GroupMetadata holds the per-group properties the scorer needs, learned from
// BlockStored events. It is engine-agnostic: the kvevents layer classifies the
// engine's cache-spec kind into these fields, so this package stays free of
// vLLM-specific vocabulary.
type GroupMetadata struct {
	// IsMainAttention marks a group whose blocks gate the contiguous prefix
	// (full / MLA / sink-full attention).
	IsMainAttention bool
	// BlockSize is the group's block size in tokens.
	BlockSize int
	// SlidingWindowSize, when non-nil, marks a sliding-window group and gives
	// its window in tokens.
	SlidingWindowSize *int
}

// GroupCatalog is a thread-safe catalog of per-pod KV cache group metadata.
type GroupCatalog struct {
	mu      sync.RWMutex
	entries map[string]map[GroupID]GroupMetadata
}

// NewGroupCatalog creates a new, empty GroupCatalog.
func NewGroupCatalog() *GroupCatalog {
	return &GroupCatalog{
		entries: make(map[string]map[GroupID]GroupMetadata),
	}
}

// Learn records group metadata for a pod.
func (c *GroupCatalog) Learn(podID string, g GroupID, meta GroupMetadata) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.entries[podID] == nil {
		c.entries[podID] = make(map[GroupID]GroupMetadata)
	}
	c.entries[podID][g] = meta
}

// IsMainGroup reports whether group g on podID is a main-attention group whose
// blocks gate prefix-cache routing.
//
// When the catalog has not yet learned the group — an event race before the
// first BlockStored, or an older vLLM that does not emit kv_cache_spec_kind —
// it falls back to treating group_idx 0 as the main group. The method is
// nil-safe: a nil catalog always uses the group_idx 0 fallback, so a scorer
// wired without a catalog still routes correctly for the common case where full
// attention is group 0.
func (c *GroupCatalog) IsMainGroup(podID string, g GroupID) bool {
	if meta, ok := c.Get(podID, g); ok {
		return meta.IsMainAttention
	}
	return g == 0
}

// SlidingWindowGroup describes a sliding-window KV cache group that the scorer
// can model precisely against the request-key block sequence.
type SlidingWindowGroup struct {
	// GroupID is the vLLM KV cache group identity.
	GroupID GroupID
	// ContiguousBlocks is the number of contiguous cached blocks a prefix-cache
	// hit requires at its trailing edge, cdiv(window-1, blockSize) — mirroring
	// vLLM's SlidingWindowManager._contiguous_blocks_for_hit.
	ContiguousBlocks int
}

// SlidingWindowGroups returns the sliding-window groups for podID, with the
// contiguous trailing-block count each needs for a hit.
//
// The count is cdiv(window-1, blockSize) using the group's own block size —
// mirroring vLLM's SlidingWindowManager._contiguous_blocks_for_hit, which
// divides by the group's spec block size. This assumes the router indexes at
// that block size, which holds for uniform-block-size models; differing-block-
// size hybrids (e.g. Gemma) are unsupported at the indexing layer (the single
// request-key granularity cannot match a differently-sized group's blocks) and
// are deferred (#336).
//
// A group qualifies only when it is a sliding-window group (SlidingWindowSize
// set) with a known block size and a window large enough to require at least one
// trailing block.
func (c *GroupCatalog) SlidingWindowGroups(podID string) []SlidingWindowGroup {
	if c == nil {
		return nil
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	var groups []SlidingWindowGroup
	for g, meta := range c.entries[podID] {
		if meta.SlidingWindowSize == nil || meta.BlockSize <= 0 {
			continue
		}
		need := cdiv(*meta.SlidingWindowSize-1, meta.BlockSize)
		if need <= 0 {
			continue
		}
		groups = append(groups, SlidingWindowGroup{GroupID: g, ContiguousBlocks: need})
	}
	return groups
}

// cdiv returns ceil(a/b) for non-negative a and positive b (mirrors vLLM's cdiv).
func cdiv(a, b int) int {
	return (a + b - 1) / b
}

// Get returns the metadata for a pod group. It is nil-safe.
func (c *GroupCatalog) Get(podID string, g GroupID) (GroupMetadata, bool) {
	if c == nil {
		return GroupMetadata{}, false
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	groups, ok := c.entries[podID]
	if !ok {
		return GroupMetadata{}, false
	}
	meta, ok := groups[g]
	return meta, ok
}
