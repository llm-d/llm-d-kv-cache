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

package server_test

import (
	"errors"
	"io"
	"log/slog"
	"math"
	"net"
	"path/filepath"
	"testing"
	"time"

	"golang.org/x/sys/unix"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvtier/policy"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvtier/pool"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvtier/protocol"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvtier/server"
)

const slotBytes = 4096

type client struct {
	t    *testing.T
	conn *net.UnixConn
	mem  []byte
	id   uint64
}

func startServer(t *testing.T, slots uint64) string {
	t.Helper()
	return startServerGrace(t, slots, 0)
}

// startServerGrace starts a server that holds a disconnected session's
// pins and half-written slots for grace.
func startServerGrace(t *testing.T, slots uint64, grace time.Duration) string {
	t.Helper()
	newPolicy := func() policy.Policy { return policy.NewLRU() }
	store := pool.NewStore(pool.ArenaOptions{CapacityBytes: slots * slotBytes}, newPolicy)
	srv := server.New(store, slog.New(slog.NewTextHandler(io.Discard, nil)))
	srv.SetReleaseGrace(grace)
	path := filepath.Join(t.TempDir(), "d.sock")
	if err := srv.Listen(path); err != nil {
		t.Fatal(err)
	}
	serve(t, srv, nil)
	return path
}

// serve runs srv until the test ends. At cleanup it calls stop, if set,
// then closes srv and checks the errors.
func serve(t *testing.T, srv *server.Server, stop func()) {
	t.Helper()
	done := make(chan error, 1)
	go func() { done <- srv.Serve() }()
	t.Cleanup(func() {
		if stop != nil {
			stop()
		}
		if err := srv.Close(); err != nil {
			t.Error(err)
		}
		if err := <-done; err != nil {
			t.Error(err)
		}
	})
}

// closeOnCleanup closes conn when the test ends.
func closeOnCleanup(t *testing.T, conn io.Closer) {
	t.Helper()
	t.Cleanup(func() {
		if err := conn.Close(); err != nil && !errors.Is(err, net.ErrClosed) {
			t.Error(err)
		}
	})
}

// encode encodes m and fails the test on error.
func encode(t *testing.T, m *protocol.Message) []byte {
	t.Helper()
	frame, err := protocol.Encode(m)
	if err != nil {
		t.Fatal(err)
	}
	return frame
}

// send encodes m and writes it to conn.
func send(t *testing.T, conn net.Conn, m *protocol.Message) {
	t.Helper()
	if _, err := conn.Write(encode(t, m)); err != nil {
		t.Fatal(err)
	}
}

func dial(t *testing.T, path, ns string) *client {
	t.Helper()
	conn, err := net.DialUnix("unix", nil, &net.UnixAddr{Name: path, Net: "unix"})
	if err != nil {
		t.Fatal(err)
	}
	closeOnCleanup(t, conn)
	cl := &client{t: t, conn: conn}
	send(t, conn, &protocol.Message{ID: 1, Op: protocol.OpHello, Version: protocol.Version, Namespace: ns, SlotBytes: slotBytes})
	buf := make([]byte, 1<<16)
	oob := make([]byte, unix.CmsgSpace(4))
	n, oobn, _, _, err := conn.ReadMsgUnix(buf, oob)
	if err != nil {
		t.Fatal(err)
	}
	msgs, err := unix.ParseSocketControlMessage(oob[:oobn])
	if err != nil || len(msgs) != 1 {
		t.Fatalf("control messages: %v %v", msgs, err)
	}
	fds, err := unix.ParseUnixRights(&msgs[0])
	if err != nil || len(fds) != 1 {
		t.Fatalf("rights: %v %v", fds, err)
	}
	resp, err := protocol.Read(bytesReader(buf[:n]))
	if err != nil || resp.Error != "" {
		t.Fatalf("hello: %v %v", resp, err)
	}
	if resp.ArenaBytes > math.MaxInt {
		t.Fatalf("arena of %d bytes is too large", resp.ArenaBytes)
	}
	cl.mem, err = unix.Mmap(fds[0], 0, int(resp.ArenaBytes), unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED)
	if err != nil {
		t.Fatal(err)
	}
	if err := unix.Close(fds[0]); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := unix.Munmap(cl.mem); err != nil {
			t.Error(err)
		}
	})
	cl.id = 1
	return cl
}

type sliceReader struct{ b []byte }

func (r *sliceReader) Read(p []byte) (int, error) {
	if len(r.b) == 0 {
		return 0, io.EOF
	}
	n := copy(p, r.b)
	r.b = r.b[n:]
	return n, nil
}

func bytesReader(b []byte) io.Reader { return &sliceReader{b} }

// call sends a request and returns its response, skipping events.
func (c *client) call(m *protocol.Message) *protocol.Message {
	c.t.Helper()
	c.id++
	m.ID = c.id
	send(c.t, c.conn, m)
	for {
		if err := c.conn.SetReadDeadline(time.Now().Add(5 * time.Second)); err != nil {
			c.t.Fatal(err)
		}
		resp, err := protocol.Read(c.conn)
		if err != nil {
			c.t.Fatal(err)
		}
		if resp.ID == m.ID {
			if resp.Error != "" {
				c.t.Fatalf("%s: %s", m.Op, resp.Error)
			}
			return resp
		}
	}
}

func (c *client) event() *protocol.Message {
	c.t.Helper()
	if err := c.conn.SetReadDeadline(time.Now().Add(5 * time.Second)); err != nil {
		c.t.Fatal(err)
	}
	m, err := protocol.Read(c.conn)
	if err != nil {
		c.t.Fatal(err)
	}
	if m.Op != protocol.OpEvent {
		c.t.Fatalf("got %s, want event", m.Op)
	}
	return m
}

func TestTwoClientsShareBlocks(t *testing.T) {
	path := startServer(t, 4)
	ca := dial(t, path, "ns")
	cb := dial(t, path, "ns")

	key := [][]byte{[]byte("block-1")}
	r := ca.call(&protocol.Message{Op: protocol.OpAlloc, Keys: key})
	if r.Status[0] != uint8(pool.Allocated) {
		t.Fatalf("alloc status %v", r.Status)
	}
	off := r.Slots[0] * slotBytes
	copy(ca.mem[off:], "kv-bytes")
	ca.call(&protocol.Message{Op: protocol.OpCommit, Keys: key, OK: []bool{true}})

	if ev := cb.event(); len(ev.Stored) != 1 || string(ev.Stored[0]) != "block-1" {
		t.Fatalf("event %+v", ev)
	}
	p := cb.call(&protocol.Message{Op: protocol.OpPin, Keys: key})
	if p.Slots[0] != r.Slots[0] {
		t.Fatalf("pin slot %d, want %d", p.Slots[0], r.Slots[0])
	}
	if got := string(cb.mem[off : off+8]); got != "kv-bytes" {
		t.Fatalf("cb read %q", got)
	}
	cb.call(&protocol.Message{Op: protocol.OpUnpin, Keys: key})

	// A client that reconnects, as an engine does after a restart, finds the
	// block in its snapshot.
	cc := dial(t, path, "ns")
	s := cc.call(&protocol.Message{Op: protocol.OpSnapshot})
	if len(s.Keys) != 1 || string(s.Keys[0]) != "block-1" {
		t.Fatalf("snapshot %v", s.Keys)
	}
}

func TestNamespacesAreIsolated(t *testing.T) {
	path := startServer(t, 4)
	ca := dial(t, path, "model-a")
	cb := dial(t, path, "model-b")

	key := [][]byte{[]byte("block-1")}
	ca.call(&protocol.Message{Op: protocol.OpAlloc, Keys: key})
	ca.call(&protocol.Message{Op: protocol.OpCommit, Keys: key, OK: []bool{true}})

	if p := cb.call(&protocol.Message{Op: protocol.OpPin, Keys: key}); p.Slots[0] != -1 {
		t.Fatalf("model-b pinned model-a's block at slot %d", p.Slots[0])
	}
	if s := cb.call(&protocol.Message{Op: protocol.OpSnapshot}); len(s.Keys) != 0 {
		t.Fatalf("model-b snapshot %v, want empty", s.Keys)
	}
	r := cb.call(&protocol.Message{Op: protocol.OpAlloc, Keys: key})
	if r.Status[0] != uint8(pool.Allocated) {
		t.Fatalf("model-b alloc status %v, want a fresh slot", r.Status)
	}
}

func TestEvictionEventsReachOtherClients(t *testing.T) {
	path := startServer(t, 1)
	a := dial(t, path, "ns")
	b := dial(t, path, "ns")
	for i, key := range []string{"k1", "k2"} {
		keys := [][]byte{[]byte(key)}
		r := a.call(&protocol.Message{Op: protocol.OpAlloc, Keys: keys})
		if i == 1 && (len(r.Removed) != 1 || string(r.Removed[0]) != "k1") {
			t.Fatalf("alloc removed %v, want k1", r.Removed)
		}
		a.call(&protocol.Message{Op: protocol.OpCommit, Keys: keys, OK: []bool{true}})
	}
	b.event() // stored k1
	if ev := b.event(); len(ev.Removed) != 1 || string(ev.Removed[0]) != "k1" {
		t.Fatalf("event %+v, want removed k1", ev)
	}
}

func TestDisconnectReleasesPins(t *testing.T) {
	path := startServerGrace(t, 1, 300*time.Millisecond)
	ca := dial(t, path, "ns")
	keys := [][]byte{[]byte("k1")}
	ca.call(&protocol.Message{Op: protocol.OpAlloc, Keys: keys})
	ca.call(&protocol.Message{Op: protocol.OpCommit, Keys: keys, OK: []bool{true}})

	cb := dial(t, path, "ns")
	cb.call(&protocol.Message{Op: protocol.OpPin, Keys: keys})
	r := ca.call(&protocol.Message{Op: protocol.OpAlloc, Keys: [][]byte{[]byte("k2")}})
	if r.Status[0] != uint8(pool.NoSpace) {
		t.Fatalf("status %v, want NoSpace while k1 is pinned", r.Status)
	}
	if err := cb.conn.Close(); err != nil {
		t.Fatal(err)
	}
	// b may still be copying out of k1's slot, so the pin holds for the
	// grace period.
	time.Sleep(100 * time.Millisecond)
	r = ca.call(&protocol.Message{Op: protocol.OpAlloc, Keys: [][]byte{[]byte("k2")}})
	if r.Status[0] != uint8(pool.NoSpace) {
		t.Fatalf("status %v, want NoSpace during the grace period", r.Status)
	}
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		r = ca.call(&protocol.Message{Op: protocol.OpAlloc, Keys: [][]byte{[]byte("k2")}})
		if r.Status[0] == uint8(pool.Allocated) {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("pin not released after disconnect")
}

func TestHelloRejectsSlotMismatch(t *testing.T) {
	path := startServer(t, 4)
	dial(t, path, "ns")
	conn, err := net.DialUnix("unix", nil, &net.UnixAddr{Name: path, Net: "unix"})
	if err != nil {
		t.Fatal(err)
	}
	closeOnCleanup(t, conn)
	hello := &protocol.Message{ID: 1, Op: protocol.OpHello, Version: protocol.Version, Namespace: "ns", SlotBytes: slotBytes * 2}
	send(t, conn, hello)
	resp, err := protocol.Read(conn)
	if err != nil || resp.Error == "" {
		t.Fatalf("expected error, got %+v %v", resp, err)
	}
}

func TestHelloAttachWithoutSlotSize(t *testing.T) {
	path := startServer(t, 4)
	conn, err := net.DialUnix("unix", nil, &net.UnixAddr{Name: path, Net: "unix"})
	if err != nil {
		t.Fatal(err)
	}
	closeOnCleanup(t, conn)
	send(t, conn, &protocol.Message{ID: 1, Op: protocol.OpHello, Version: protocol.Version, Namespace: "ns"})
	if resp, err := protocol.Read(conn); err != nil || resp.Error == "" {
		t.Fatalf("attach before any pool: %+v %v", resp, err)
	}
	dial(t, path, "ns") // creates the pool
	c := &client{t: t}
	conn2, err := net.DialUnix("unix", nil, &net.UnixAddr{Name: path, Net: "unix"})
	if err != nil {
		t.Fatal(err)
	}
	closeOnCleanup(t, conn2)
	c.conn = conn2
	send(t, conn2, &protocol.Message{ID: 1, Op: protocol.OpHello, Version: protocol.Version, Namespace: "ns"})
	buf := make([]byte, 1<<16)
	oob := make([]byte, unix.CmsgSpace(4))
	n, oobn, _, _, err := conn2.ReadMsgUnix(buf, oob)
	if err != nil || oobn == 0 {
		t.Fatalf("attach: %v oob=%d", err, oobn)
	}
	resp, err := protocol.Read(bytesReader(buf[:n]))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Error != "" || resp.SlotBytes != slotBytes {
		t.Fatalf("attach response %+v", resp)
	}
}

func TestDisconnectHoldsHalfWrittenSlots(t *testing.T) {
	path := startServerGrace(t, 1, 300*time.Millisecond)
	a := dial(t, path, "ns")
	a.call(&protocol.Message{Op: protocol.OpAlloc, Keys: [][]byte{[]byte("k1")}})
	if err := a.conn.Close(); err != nil {
		t.Fatal(err)
	}
	b := dial(t, path, "ns")
	time.Sleep(100 * time.Millisecond)
	// a may still be copying into the slot it reserved.
	if r := b.call(&protocol.Message{Op: protocol.OpAlloc, Keys: [][]byte{[]byte("k2")}}); r.Status[0] != uint8(pool.NoSpace) {
		t.Fatalf("status %v, want NoSpace during the grace period", r.Status)
	}
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		r := b.call(&protocol.Message{Op: protocol.OpAlloc, Keys: [][]byte{[]byte("k2")}})
		if r.Status[0] == uint8(pool.Allocated) {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("slot not released after the grace period")
}

func TestEventsPrecedeTheirReply(t *testing.T) {
	path := startServer(t, 1)
	a := dial(t, path, "ns")
	keys := [][]byte{[]byte("k1")}
	a.call(&protocol.Message{Op: protocol.OpAlloc, Keys: keys})
	a.id++
	send(t, a.conn, &protocol.Message{ID: a.id, Op: protocol.OpCommit, Keys: keys, OK: []bool{true}})
	// The caller gets its own stored event, and before the commit reply, so
	// an event stream alone keeps a client's view in order.
	if ev := a.event(); len(ev.Stored) != 1 || string(ev.Stored[0]) != "k1" {
		t.Fatalf("event %+v, want stored k1", ev)
	}
	if err := a.conn.SetReadDeadline(time.Now().Add(5 * time.Second)); err != nil {
		t.Fatal(err)
	}
	if r, err := protocol.Read(a.conn); err != nil || r.ID != a.id {
		t.Fatalf("got %+v %v, want the commit reply", r, err)
	}
}

func TestCloseDropsConnectionsBeforeHello(t *testing.T) {
	newPolicy := func() policy.Policy { return policy.NewLRU() }
	store := pool.NewStore(pool.ArenaOptions{CapacityBytes: slotBytes}, newPolicy)
	srv := server.New(store, slog.New(slog.NewTextHandler(io.Discard, nil)))
	path := filepath.Join(t.TempDir(), "d.sock")
	if err := srv.Listen(path); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- srv.Serve() }()
	conn, err := net.DialUnix("unix", nil, &net.UnixAddr{Name: path, Net: "unix"})
	if err != nil {
		t.Fatal(err)
	}
	closeOnCleanup(t, conn)
	// The connection never sends hello.
	closed := make(chan error, 1)
	go func() { closed <- srv.Close() }()
	select {
	case err := <-closed:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Close hung on a connection that never sent hello")
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}
