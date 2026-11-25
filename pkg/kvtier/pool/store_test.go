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

package pool_test

import (
	"testing"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvtier/policy"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvtier/pool"
)

const slot = 4096

func newStore(t *testing.T, slots uint64) *pool.Store {
	t.Helper()
	s := pool.NewStore(pool.ArenaOptions{CapacityBytes: slots * slot}, func() policy.Policy { return policy.NewLRU() })
	arena, err := s.EnsureArena(slot)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := arena.Close(); err != nil {
			t.Error(err)
		}
	})
	return s
}

// alloc calls Alloc and fails the test on error. It returns the slots and
// their status.
func alloc(t *testing.T, s *pool.Store, sid pool.SessionID, keys ...[]byte) ([]int64, []pool.AllocStatus) {
	t.Helper()
	slots, status, _, err := s.Alloc(sid, "ns", keys)
	if err != nil {
		t.Fatal(err)
	}
	return slots, status
}

func k(n byte) []byte { return []byte{n} }

// store allocates and commits keys in session 1.
func store(t *testing.T, s *pool.Store, keys ...[]byte) []pool.KeyRef {
	t.Helper()
	const sid = 1
	_, status, evicted, err := s.Alloc(sid, "ns", keys)
	if err != nil {
		t.Fatal(err)
	}
	ok := make([]bool, len(keys))
	for i, st := range status {
		ok[i] = st == pool.Allocated
	}
	if _, err := s.Commit(sid, "ns", keys, ok); err != nil {
		t.Fatal(err)
	}
	return evicted
}

func TestStoreLoadRoundTrip(t *testing.T) {
	s := newStore(t, 4)
	slots, status, _, err := s.Alloc(1, "ns", [][]byte{k(1)})
	if err != nil || status[0] != pool.Allocated {
		t.Fatalf("alloc: %v %v", status, err)
	}
	copy(s.Arena().Slot(slots[0]), "hello")
	if got := s.Pin(2, "ns", [][]byte{k(1)}); got[0] != -1 {
		t.Fatalf("block visible before commit: slot %d", got[0])
	}
	if _, err := s.Commit(1, "ns", [][]byte{k(1)}, []bool{true}); err != nil {
		t.Fatal(err)
	}
	got := s.Pin(2, "ns", [][]byte{k(1)})
	if got[0] != slots[0] || string(s.Arena().Slot(got[0])[:5]) != "hello" {
		t.Fatalf("pin returned slot %d", got[0])
	}
	if _, status := alloc(t, s, 1, k(1)); status[0] != pool.Present {
		t.Fatalf("second alloc status %d, want Present", status[0])
	}
}

func TestNamespacesAreSeparate(t *testing.T) {
	s := newStore(t, 4)
	store(t, s, k(1))
	if got := s.Pin(1, "other", [][]byte{k(1)}); got[0] != -1 {
		t.Fatal("key leaked across namespaces")
	}
}

func TestLRUEvictionSkipsPinned(t *testing.T) {
	s := newStore(t, 2)
	store(t, s, k(1))
	store(t, s, k(2))
	s.Pin(9, "ns", [][]byte{k(1)}) // k1 is now most recent and pinned
	evicted := store(t, s, k(3))
	if len(evicted) != 1 || evicted[0].Key[0] != 2 {
		t.Fatalf("evicted %v, want key 2", evicted)
	}
	// Both remaining blocks pinned: nothing can be evicted.
	s.Pin(9, "ns", [][]byte{k(3)})
	_, status := alloc(t, s, 1, k(4))
	if status[0] != pool.NoSpace {
		t.Fatalf("status %d, want NoSpace", status[0])
	}
	s.Unpin(9, "ns", [][]byte{k(1)})
	if evicted := store(t, s, k(4)); len(evicted) != 1 || evicted[0].Key[0] != 1 {
		t.Fatalf("evicted %v, want key 1", evicted)
	}
}

func TestFailedCommitFreesSlot(t *testing.T) {
	s := newStore(t, 1)
	alloc(t, s, 1, k(1))
	if _, err := s.Commit(1, "ns", [][]byte{k(1)}, []bool{false}); err != nil {
		t.Fatal(err)
	}
	if _, status := alloc(t, s, 1, k(2)); status[0] != pool.Allocated {
		t.Fatalf("slot not freed: status %d", status[0])
	}
}

func TestCommitIgnoresOtherSessionsWrites(t *testing.T) {
	s := newStore(t, 2)
	alloc(t, s, 1, k(1))
	stored, err := s.Commit(2, "ns", [][]byte{k(1)}, []bool{true})
	if err != nil {
		t.Fatal(err)
	}
	if len(stored) != 0 {
		t.Fatal("session 2 committed session 1's write")
	}
}

func TestReleaseSession(t *testing.T) {
	s := newStore(t, 2)
	store(t, s, k(1))
	s.Pin(2, "ns", [][]byte{k(1)})
	alloc(t, s, 2, k(2)) // left half-written
	s.ReleaseSession(2)
	st := s.Stats()
	if st.PinnedSlots != 0 || st.UsedSlots != 1 || st.ReadySlots != 1 {
		t.Fatalf("after release: %+v", st)
	}
}

func TestSlotSizeMismatch(t *testing.T) {
	s := newStore(t, 2)
	if _, err := s.EnsureArena(slot * 2); err == nil {
		t.Fatal("expected slot size mismatch")
	}
}
