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
	// BlockSize is the group's engine block size in tokens. Diagnostic only:
	// scoring runs at the router's canonical request-key granularity, so the
	// engine block size never enters the scoring math.
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

// hasMainGroupLocked reports whether podID has a main-attention group,
// mirroring IsMainGroup's fallback: an unlearned pod or an unlearned group 0
// reports true, so only pods positively known to lack main attention (e.g.
// SWA-only models) report false. Keeping the fallback consistent with
// IsMainGroup ensures a pod is never claimed by both the main-prefix scoring
// path and the no-main-group fallback path. Caller must hold c.mu.
func (c *GroupCatalog) hasMainGroupLocked(podID string) bool {
	groups := c.entries[podID]
	if len(groups) == 0 {
		return true
	}
	if _, ok := groups[0]; !ok {
		// Group 0 unlearned: IsMainGroup's fallback would treat its entries as
		// main, so the pod belongs to the main-prefix path.
		return true
	}
	for _, meta := range groups {
		if meta.IsMainAttention {
			return true
		}
	}
	return false
}

// PodsWithoutMainGroup returns the pods positively known to have no
// main-attention group. These run SWA-only models, which vLLM serves through
// its unitary coordinator: the trailing-window cache-hit scan runs over the
// whole request rather than being bounded by a full-attention prefix. It is
// nil-safe (a nil catalog knows no such pods).
func (c *GroupCatalog) PodsWithoutMainGroup() []string {
	if c == nil {
		return nil
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	var pods []string
	for podID := range c.entries {
		if !c.hasMainGroupLocked(podID) {
			pods = append(pods, podID)
		}
	}
	return pods
}

// SlidingWindowClass groups a pod's sliding-window KV cache groups that share
// one trailing-window requirement. vLLM scans same-spec groups jointly — a
// block counts only when cached in every group of the spec
// (BlockPool.get_cached_block misses if any group misses) — so the scorer
// scans each class once with AND-presence across its groups.
type SlidingWindowClass struct {
	// GroupIDs are the KV cache groups sharing this trailing-window requirement.
	GroupIDs []GroupID
	// ContiguousBlocks is the number of contiguous cached request-key blocks a
	// prefix-cache hit requires at its trailing edge: cdiv(window-1,
	// canonicalBlockSize) — vLLM's
	// SlidingWindowManager._contiguous_blocks_for_hit in router units.
	ContiguousBlocks int
}

// SlidingWindowClasses returns podID's sliding-window groups bucketed by their
// trailing-window requirement in canonical request-key blocks.
//
// The count is cdiv(window-1, canonicalBlockSize): the scorer scans canonical
// request keys, so the window (a token count) converts to key counts with the
// router's own block size. The group's engine block size is irrelevant here —
// presence at a canonical key is derived from the group's events re-chunked at
// canonical granularity, whatever the engine granularity.
//
// A group qualifies only when it is a sliding-window group (SlidingWindowSize
// set) with a window large enough to require at least one trailing block.
func (c *GroupCatalog) SlidingWindowClasses(podID string, canonicalBlockSize int) []SlidingWindowClass {
	if c == nil || canonicalBlockSize <= 0 {
		return nil
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	var classes []SlidingWindowClass
	for g, meta := range c.entries[podID] {
		if meta.SlidingWindowSize == nil {
			continue
		}
		need := cdiv(*meta.SlidingWindowSize-1, canonicalBlockSize)
		if need <= 0 {
			continue
		}
		idx := -1
		for i := range classes {
			if classes[i].ContiguousBlocks == need {
				idx = i
				break
			}
		}
		if idx == -1 {
			classes = append(classes, SlidingWindowClass{ContiguousBlocks: need})
			idx = len(classes) - 1
		}
		classes[idx].GroupIDs = append(classes[idx].GroupIDs, g)
	}
	return classes
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
