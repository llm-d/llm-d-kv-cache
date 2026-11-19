// Copyright 2026 The llm-d Authors.
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

// Package policy defines how the daemon picks which stored block to evict.
package policy

import "container/list"

// Policy orders committed blocks for eviction. Entries are identified by
// their slot index, which is unique among live entries.
//
// Implementations are not safe for concurrent use; the caller holds the
// store lock.
type Policy interface {
	// Insert adds a newly committed entry.
	Insert(id int64)
	// Touch records a use of an entry (a load).
	Touch(id int64)
	// Remove drops an entry that was evicted or freed.
	Remove(id int64)
	// Victim returns the next entry to evict. Entries for which skip returns
	// true (for example, pinned entries) are passed over.
	Victim(skip func(id int64) bool) (int64, bool)
	// Len returns the number of tracked entries.
	Len() int
}

// New returns the policy registered under name. It returns nil for an
// unknown name.
func New(name string) Policy {
	switch name {
	case "", "lru":
		return NewLRU()
	default:
		return nil
	}
}

// LRU evicts the least recently stored or loaded entry.
type LRU struct {
	order *list.List // front = most recently used
	elems map[int64]*list.Element
}

// NewLRU returns an empty LRU policy.
func NewLRU() *LRU {
	return &LRU{order: list.New(), elems: make(map[int64]*list.Element)}
}

func (l *LRU) Insert(id int64) {
	if e, ok := l.elems[id]; ok {
		l.order.MoveToFront(e)
		return
	}
	l.elems[id] = l.order.PushFront(id)
}

func (l *LRU) Touch(id int64) {
	if e, ok := l.elems[id]; ok {
		l.order.MoveToFront(e)
	}
}

func (l *LRU) Remove(id int64) {
	if e, ok := l.elems[id]; ok {
		l.order.Remove(e)
		delete(l.elems, id)
	}
}

func (l *LRU) Victim(skip func(id int64) bool) (int64, bool) {
	for e := l.order.Back(); e != nil; e = e.Prev() {
		id, ok := e.Value.(int64)
		if !ok {
			continue
		}
		if skip != nil && skip(id) {
			continue
		}
		return id, true
	}
	return 0, false
}

func (l *LRU) Len() int { return len(l.elems) }
