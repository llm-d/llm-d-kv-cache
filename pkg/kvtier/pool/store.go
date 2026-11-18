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

// Package pool holds the daemon's KV blocks: the shared-memory arena, the
// index from block key to slot, pins, and eviction.
package pool

import (
	"errors"
	"fmt"
	"sync"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvtier/policy"
)

// SessionID identifies one connected engine-side client.
type SessionID uint64

// AllocStatus is the per-key outcome of Alloc.
type AllocStatus uint8

const (
	// Allocated means a slot was reserved; the client must write it and
	// then Commit.
	Allocated AllocStatus = 0
	// Present means the block is already stored or being written.
	Present AllocStatus = 1
	// NoSpace means every slot is in use and none could be evicted.
	NoSpace AllocStatus = 2
)

type state uint8

const (
	writing state = iota + 1
	ready
)

// KeyRef names a block within a namespace.
type KeyRef struct {
	Namespace string
	Key       []byte
}

type entry struct {
	ns    string
	key   []byte
	slot  int64
	state state
	owner SessionID         // writer while state == writing
	pins  map[SessionID]int // loads in progress
}

func (e *entry) pinned() bool { return len(e.pins) > 0 }

// Stats are counters exposed on /metrics.
type Stats struct {
	SlotBytes    uint64
	NumSlots     int64
	UsedSlots    int64
	ReadySlots   int64
	PinnedSlots  int64
	Stores       uint64
	StoreNoSpace uint64
	Loads        uint64
	LoadMisses   uint64
	Evictions    uint64
}

// Store is the daemon's block index over one arena. All methods are safe
// for concurrent use.
type Store struct {
	mu       sync.Mutex
	opts     ArenaOptions
	newPol   func() policy.Policy
	arena    *Arena
	free     []int64
	bySlot   []*entry
	byKey    map[string]*entry
	pol      policy.Policy
	sessions map[SessionID]map[string]struct{} // keys pinned or written by a session
	stats    Stats
}

// ErrSlotSize is returned when a client's block size differs from the
// arena's slot size.
var ErrSlotSize = errors.New("block size does not match the daemon's slot size")

// NewStore returns a store. The arena is created on first use, because the
// slot size comes from the first client unless opts.SlotBytes is set.
func NewStore(opts ArenaOptions, newPolicy func() policy.Policy) *Store {
	return &Store{
		opts:     opts,
		newPol:   newPolicy,
		byKey:    make(map[string]*entry),
		sessions: make(map[SessionID]map[string]struct{}),
	}
}

// EnsureArena creates the arena for slotBytes, or checks that the existing
// arena uses that slot size.
func (s *Store) EnsureArena(slotBytes uint64) (*Arena, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.arena != nil {
		if s.arena.SlotBytes() != slotBytes {
			return nil, fmt.Errorf("%w: daemon %d, client %d", ErrSlotSize, s.arena.SlotBytes(), slotBytes)
		}
		return s.arena, nil
	}
	if s.opts.SlotBytes != 0 && s.opts.SlotBytes != slotBytes {
		return nil, fmt.Errorf("%w: daemon %d, client %d", ErrSlotSize, s.opts.SlotBytes, slotBytes)
	}
	opts := s.opts
	opts.SlotBytes = slotBytes
	a, err := NewArena(opts)
	if err != nil {
		return nil, err
	}
	s.arena = a
	s.pol = s.newPol()
	s.bySlot = make([]*entry, a.NumSlots())
	s.free = make([]int64, 0, a.NumSlots())
	for i := a.NumSlots() - 1; i >= 0; i-- {
		s.free = append(s.free, i)
	}
	s.stats.SlotBytes = slotBytes
	s.stats.NumSlots = a.NumSlots()
	return a, nil
}

// Arena returns the arena, or nil before the first client.
func (s *Store) Arena() *Arena {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.arena
}

func mapKey(ns string, key []byte) string {
	return ns + "\x00" + string(key)
}

func (s *Store) track(sid SessionID, mk string) {
	m := s.sessions[sid]
	if m == nil {
		m = make(map[string]struct{})
		s.sessions[sid] = m
	}
	m[mk] = struct{}{}
}

func (s *Store) untrack(sid SessionID, mk string) {
	if m := s.sessions[sid]; m != nil {
		delete(m, mk)
	}
}

// takeSlot returns a free slot, evicting the policy's victim when none is
// free. Evicted blocks are appended to evicted.
func (s *Store) takeSlot(evicted *[]KeyRef) (int64, bool) {
	if n := len(s.free); n > 0 {
		slot := s.free[n-1]
		s.free = s.free[:n-1]
		return slot, true
	}
	victim, ok := s.pol.Victim(func(id int64) bool {
		e := s.bySlot[id]
		return e == nil || e.state != ready || e.pinned()
	})
	if !ok {
		return 0, false
	}
	e := s.bySlot[victim]
	s.pol.Remove(victim)
	delete(s.byKey, mapKey(e.ns, e.key))
	s.bySlot[victim] = nil
	s.stats.Evictions++
	*evicted = append(*evicted, KeyRef{Namespace: e.ns, Key: e.key})
	return victim, true
}

func (s *Store) freeSlot(slot int64) {
	s.bySlot[slot] = nil
	s.free = append(s.free, slot)
}

// Alloc reserves a slot for each key that is not already present. The
// caller writes each Allocated slot and then calls Commit. Blocks evicted to
// make room are returned so the caller can announce them.
func (s *Store) Alloc(sid SessionID, ns string, keys [][]byte) ([]int64, []AllocStatus, []KeyRef, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.arena == nil {
		return nil, nil, nil, errors.New("arena not initialised")
	}
	slots := make([]int64, len(keys))
	status := make([]AllocStatus, len(keys))
	var evicted []KeyRef
	for i, k := range keys {
		slots[i] = -1
		mk := mapKey(ns, k)
		if _, ok := s.byKey[mk]; ok {
			status[i] = Present
			continue
		}
		slot, ok := s.takeSlot(&evicted)
		if !ok {
			status[i] = NoSpace
			s.stats.StoreNoSpace++
			continue
		}
		e := &entry{ns: ns, key: append([]byte(nil), k...), slot: slot, state: writing, owner: sid}
		s.bySlot[slot] = e
		s.byKey[mk] = e
		s.track(sid, mk)
		slots[i] = slot
		status[i] = Allocated
	}
	return slots, status, evicted, nil
}

// Commit finishes writes started by Alloc. A key with ok[i] true becomes
// loadable. A key with ok[i] false has its slot freed. Keys that this
// session did not allocate are ignored. Returns the blocks made loadable.
func (s *Store) Commit(sid SessionID, ns string, keys [][]byte, ok []bool) ([]KeyRef, error) {
	if len(keys) != len(ok) {
		return nil, fmt.Errorf("commit: %d keys but %d flags", len(keys), len(ok))
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	var stored []KeyRef
	for i, k := range keys {
		mk := mapKey(ns, k)
		e, found := s.byKey[mk]
		if !found || e.state != writing || e.owner != sid {
			continue
		}
		s.untrack(sid, mk)
		if !ok[i] {
			delete(s.byKey, mk)
			s.freeSlot(e.slot)
			continue
		}
		e.state = ready
		e.owner = 0
		s.pol.Insert(e.slot)
		s.stats.Stores++
		stored = append(stored, KeyRef{Namespace: ns, Key: e.key})
	}
	return stored, nil
}

// Pin returns the slot of each loadable key and keeps it from eviction
// until Unpin. A key that is absent or still being written gets -1.
func (s *Store) Pin(sid SessionID, ns string, keys [][]byte) []int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	slots := make([]int64, len(keys))
	for i, k := range keys {
		mk := mapKey(ns, k)
		e, ok := s.byKey[mk]
		if !ok || e.state != ready {
			slots[i] = -1
			s.stats.LoadMisses++
			continue
		}
		if e.pins == nil {
			e.pins = make(map[SessionID]int)
		}
		e.pins[sid]++
		s.track(sid, mk)
		s.pol.Touch(e.slot)
		s.stats.Loads++
		slots[i] = e.slot
	}
	return slots
}

// Unpin releases pins taken by Pin.
func (s *Store) Unpin(sid SessionID, ns string, keys [][]byte) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, k := range keys {
		mk := mapKey(ns, k)
		e, ok := s.byKey[mk]
		if !ok || e.pins[sid] == 0 {
			continue
		}
		e.pins[sid]--
		if e.pins[sid] == 0 {
			delete(e.pins, sid)
			s.untrack(sid, mk)
		}
	}
}

// Snapshot returns the keys of every loadable block in a namespace.
func (s *Store) Snapshot(ns string) [][]byte {
	s.mu.Lock()
	defer s.mu.Unlock()
	var keys [][]byte
	for _, e := range s.byKey {
		if e.ns == ns && e.state == ready {
			keys = append(keys, e.key)
		}
	}
	return keys
}

// ReleaseSession drops every pin a session holds and frees slots it was
// still writing. Call it when a client disconnects.
func (s *Store) ReleaseSession(sid SessionID) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for mk := range s.sessions[sid] {
		e, ok := s.byKey[mk]
		if !ok {
			continue
		}
		if e.state == writing && e.owner == sid {
			delete(s.byKey, mk)
			s.freeSlot(e.slot)
			continue
		}
		delete(e.pins, sid)
	}
	delete(s.sessions, sid)
}

// Stats returns a snapshot of the counters.
func (s *Store) Stats() Stats {
	s.mu.Lock()
	defer s.mu.Unlock()
	st := s.stats
	for _, e := range s.bySlot {
		if e == nil {
			continue
		}
		st.UsedSlots++
		if e.state == ready {
			st.ReadySlots++
		}
		if e.pinned() {
			st.PinnedSlots++
		}
	}
	return st
}
