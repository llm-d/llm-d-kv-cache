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

package pool

import (
	"fmt"
	"math"

	"golang.org/x/sys/unix"
)

const hugePageSize = 2 << 20

// Arena is a shared-memory region carved into fixed-size slots. The daemon
// owns it. Engine-side shims receive its file descriptor and map it, so KV
// blocks survive an engine restart.
type Arena struct {
	fd       int
	size     uint64
	data     []byte
	slotSize uint64
	numSlots int64
}

// ArenaOptions configure NewArena.
type ArenaOptions struct {
	// CapacityBytes is the most memory the arena may use.
	CapacityBytes uint64
	// SlotBytes is the size of one block (one vLLM offload chunk).
	SlotBytes uint64
	// HugeTLB backs the arena with 2 MiB huge pages.
	HugeTLB bool
	// Prefault touches every page at creation. Pages of a memfd are charged
	// to the memory cgroup of the process that first faults them in, so
	// prefaulting charges the arena to the daemon's container, not the
	// engine's.
	Prefault bool
}

// NewArena creates the shared-memory region.
func NewArena(opts ArenaOptions) (*Arena, error) {
	if opts.SlotBytes == 0 {
		return nil, fmt.Errorf("slot size must be positive")
	}
	numSlots := opts.CapacityBytes / opts.SlotBytes
	if numSlots == 0 {
		return nil, fmt.Errorf("capacity %d holds no slot of %d bytes", opts.CapacityBytes, opts.SlotBytes)
	}
	size := numSlots * opts.SlotBytes
	flags := unix.MFD_CLOEXEC
	if opts.HugeTLB {
		flags |= unix.MFD_HUGETLB
		size = (size + hugePageSize - 1) / hugePageSize * hugePageSize
	}
	if size > math.MaxInt64 {
		return nil, fmt.Errorf("arena of %d bytes is too large", size)
	}
	fd, err := unix.MemfdCreate("llmd-kv-tiering", flags)
	if err != nil {
		return nil, fmt.Errorf("memfd_create: %w", err)
	}
	if err := unix.Ftruncate(fd, int64(size)); err != nil {
		unix.Close(fd)
		return nil, fmt.Errorf("ftruncate %d bytes: %w", size, err)
	}
	mflags := unix.MAP_SHARED
	if opts.Prefault {
		mflags |= unix.MAP_POPULATE
	}
	data, err := unix.Mmap(fd, 0, int(size), unix.PROT_READ|unix.PROT_WRITE, mflags)
	if err != nil {
		unix.Close(fd)
		return nil, fmt.Errorf("mmap %d bytes: %w", size, err)
	}
	slotCount := int64(numSlots) // #nosec G115 -- numSlots <= size, which is checked against MaxInt64 above
	return &Arena{fd: fd, size: size, data: data, slotSize: opts.SlotBytes, numSlots: slotCount}, nil
}

// FD returns the memfd to pass to clients.
func (a *Arena) FD() int { return a.fd }

// Size returns the mapped size in bytes.
func (a *Arena) Size() uint64 { return a.size }

// SlotBytes returns the size of one slot.
func (a *Arena) SlotBytes() uint64 { return a.slotSize }

// NumSlots returns the number of slots.
func (a *Arena) NumSlots() int64 { return a.numSlots }

// Slot returns the bytes of one slot.
func (a *Arena) Slot(i int64) []byte {
	off := uint64(i) * a.slotSize // #nosec G115 -- slot indexes are in [0, numSlots)
	return a.data[off : off+a.slotSize]
}

// Close unmaps and closes the arena.
func (a *Arena) Close() error {
	err := unix.Munmap(a.data)
	if cerr := unix.Close(a.fd); err == nil {
		err = cerr
	}
	return err
}
