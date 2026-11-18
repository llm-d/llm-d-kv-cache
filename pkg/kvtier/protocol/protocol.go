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

// Package protocol is the wire format between the daemon and engine-side
// shims. Each frame is a 4-byte big-endian length followed by one msgpack
// map. The daemon passes the arena's file descriptor with SCM_RIGHTS on the
// hello response.
package protocol

import (
	"encoding/binary"
	"fmt"
	"io"

	"github.com/vmihailenco/msgpack/v5"
)

// Version is the protocol version a client must send in hello.
const Version = 1

// MaxFrame bounds a frame's size.
const MaxFrame = 256 << 20

// Operations. A request carries a nonzero ID that the response echoes.
// Events pushed by the daemon carry ID 0.
const (
	OpHello    = "hello"
	OpAlloc    = "alloc"
	OpCommit   = "commit"
	OpPin      = "pin"
	OpUnpin    = "unpin"
	OpSnapshot = "snapshot"
	OpStats    = "stats"
	OpEvent    = "event"
)

// Message is the single envelope for every request, response and event.
// Each operation uses a subset of the fields.
type Message struct {
	ID    uint64 `msgpack:"id"`
	Op    string `msgpack:"op"`
	Error string `msgpack:"error,omitempty"`

	// hello request
	Version   int    `msgpack:"version,omitempty"`
	Namespace string `msgpack:"ns,omitempty"`
	SlotBytes uint64 `msgpack:"slot_bytes,omitempty"`
	Client    string `msgpack:"client,omitempty"`
	// hello response (the arena fd travels as SCM_RIGHTS)
	ArenaBytes uint64 `msgpack:"arena_bytes,omitempty"`
	NumSlots   int64  `msgpack:"num_slots,omitempty"`
	Session    uint64 `msgpack:"session,omitempty"`

	// alloc, commit, pin, unpin, snapshot
	Keys   [][]byte `msgpack:"keys,omitempty"`
	Slots  []int64  `msgpack:"slots,omitempty"`
	Status []uint8  `msgpack:"status,omitempty"`
	OK     []bool   `msgpack:"ok,omitempty"`

	// event, and the evictions caused by alloc
	Stored  [][]byte `msgpack:"stored,omitempty"`
	Removed [][]byte `msgpack:"removed,omitempty"`

	// stats
	Stats map[string]any `msgpack:"stats,omitempty"`
}

// Encode returns m as a length-prefixed frame.
func Encode(m *Message) ([]byte, error) {
	body, err := msgpack.Marshal(m)
	if err != nil {
		return nil, err
	}
	if len(body) > MaxFrame {
		return nil, fmt.Errorf("frame of %d bytes exceeds limit", len(body))
	}
	buf := make([]byte, 4+len(body))
	binary.BigEndian.PutUint32(buf, uint32(len(body))) // #nosec G115 -- len(body) <= MaxFrame, checked above
	copy(buf[4:], body)
	return buf, nil
}

// Read reads one frame from r.
func Read(r io.Reader) (*Message, error) {
	var hdr [4]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return nil, err
	}
	n := binary.BigEndian.Uint32(hdr[:])
	if n > MaxFrame {
		return nil, fmt.Errorf("frame of %d bytes exceeds limit", n)
	}
	body := make([]byte, n)
	if _, err := io.ReadFull(r, body); err != nil {
		return nil, err
	}
	m := &Message{}
	if err := msgpack.Unmarshal(body, m); err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	return m, nil
}
