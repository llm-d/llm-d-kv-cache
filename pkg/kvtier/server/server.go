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

// Package server serves the daemon's Unix socket.
package server

import (
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/sys/unix"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvtier/pool"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvtier/protocol"
)

// outQueue bounds the frames waiting for a slow client. A client that falls
// this far behind is disconnected; it reconnects and resynchronises from a
// snapshot.
const outQueue = 4096

// helloTimeout bounds how long a new connection may take to send hello.
const helloTimeout = 30 * time.Second

// DefaultReleaseGrace is how long a disconnected session's pins and
// half-written slots are kept before reuse. A client can still be copying
// into or out of those slots when its connection drops, for example when
// the daemon closes a client that fell behind. Holding the slots for a
// while keeps a late copy from touching a block another client now owns.
const DefaultReleaseGrace = 10 * time.Second

// Server accepts shim connections and applies their requests to a Store.
type Server struct {
	store *pool.Store
	log   *slog.Logger

	nextID   atomic.Uint64
	mu       sync.Mutex
	sessions map[pool.SessionID]*session
	conns    map[*net.UnixConn]struct{} // every open connection, from accept
	closed   bool
	ln       *net.UnixListener
	wg       sync.WaitGroup

	// order serialises each pool change with the events and the reply it
	// produces, so every session sees changes in the order they happened.
	order sync.Mutex

	grace    time.Duration
	releases sync.WaitGroup
	stop     chan struct{}
}

type session struct {
	id     pool.SessionID
	conn   *net.UnixConn
	ns     string
	client string
	out    chan []byte
	once   sync.Once
	done   chan struct{}
}

func (s *session) close() {
	s.once.Do(func() {
		close(s.done)
		s.conn.Close()
	})
}

// send queues a frame. It returns false, and closes the session, when the
// queue is full.
func (s *session) send(frame []byte) bool {
	select {
	case <-s.done:
		return false
	default:
	}
	select {
	case s.out <- frame:
		return true
	default:
		s.close()
		return false
	}
}

// New returns a server over store.
func New(store *pool.Store, log *slog.Logger) *Server {
	return &Server{
		store: store, log: log,
		sessions: make(map[pool.SessionID]*session),
		conns:    make(map[*net.UnixConn]struct{}),
		grace:    DefaultReleaseGrace,
		stop:     make(chan struct{}),
	}
}

// SetReleaseGrace sets how long a disconnected session's pins and
// half-written slots are held before reuse. Call it before Serve.
func (s *Server) SetReleaseGrace(d time.Duration) { s.grace = d }

// Listen binds the Unix socket at path, replacing a stale socket file.
func (s *Server) Listen(path string) error {
	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("remove stale socket: %w", err)
	}
	ln, err := net.ListenUnix("unix", &net.UnixAddr{Name: path, Net: "unix"})
	if err != nil {
		return err
	}
	if err := os.Chmod(path, 0o660); err != nil {
		ln.Close()
		return err
	}
	s.ln = ln
	return nil
}

// Serve accepts connections until Close.
func (s *Server) Serve() error {
	for {
		conn, err := s.ln.AcceptUnix()
		if err != nil {
			if errors.Is(err, net.ErrClosed) {
				return nil
			}
			return err
		}
		s.mu.Lock()
		if s.closed {
			s.mu.Unlock()
			conn.Close()
			return nil
		}
		s.conns[conn] = struct{}{}
		s.wg.Add(1)
		s.mu.Unlock()
		go func() {
			defer s.wg.Done()
			defer func() {
				s.mu.Lock()
				delete(s.conns, conn)
				s.mu.Unlock()
			}()
			s.handle(conn)
		}()
	}
}

// Close stops accepting, closes every connection, including ones that have
// not finished hello, and waits for them. Pending releases run at once.
func (s *Server) Close() error {
	var err error
	if s.ln != nil {
		err = s.ln.Close()
	}
	s.mu.Lock()
	if !s.closed {
		s.closed = true
		close(s.stop)
	}
	for _, ss := range s.sessions {
		ss.close()
	}
	for c := range s.conns {
		c.Close()
	}
	s.mu.Unlock()
	s.wg.Wait()
	s.releases.Wait()
	return err
}

// release drops a disconnected session's pins and half-written slots after
// the grace period.
func (s *Server) release(id pool.SessionID) {
	if s.grace <= 0 || !s.store.HoldsAny(id) {
		s.store.ReleaseSession(id)
		return
	}
	s.releases.Add(1)
	go func() {
		defer s.releases.Done()
		t := time.NewTimer(s.grace)
		defer t.Stop()
		select {
		case <-t.C:
		case <-s.stop:
		}
		s.store.ReleaseSession(id)
	}()
}

func (s *Server) handle(conn *net.UnixConn) {
	ss := &session{
		id:   pool.SessionID(s.nextID.Add(1)),
		conn: conn,
		out:  make(chan []byte, outQueue),
		done: make(chan struct{}),
	}
	defer ss.close()

	if err := s.hello(ss); err != nil {
		if !errors.Is(err, io.EOF) {
			s.log.Warn("hello failed", "session", ss.id, "err", err)
		}
		return
	}

	s.mu.Lock()
	s.sessions[ss.id] = ss
	s.mu.Unlock()
	s.log.Info("client connected", "session", ss.id, "client", ss.client, "ns", ss.ns)
	defer func() {
		s.mu.Lock()
		delete(s.sessions, ss.id)
		s.mu.Unlock()
		s.release(ss.id)
		s.log.Info("client disconnected", "session", ss.id, "client", ss.client)
	}()

	go s.writer(ss)
	for {
		req, err := protocol.Read(conn)
		if err != nil {
			if !errors.Is(err, io.EOF) && !errors.Is(err, net.ErrClosed) {
				s.log.Warn("read failed", "session", ss.id, "err", err)
			}
			return
		}
		if !s.dispatchAndReply(ss, req) {
			return
		}
	}
}

func (s *Server) reply(ss *session, resp *protocol.Message) bool {
	frame, err := protocol.Encode(resp)
	if err != nil {
		s.log.Error("encode failed", "session", ss.id, "err", err)
		ss.close()
		return false
	}
	return ss.send(frame)
}

func (s *Server) writer(ss *session) {
	for {
		select {
		case <-ss.done:
			return
		case frame := <-ss.out:
			if _, err := ss.conn.Write(frame); err != nil {
				ss.close()
				return
			}
		}
	}
}

// hello reads the first request, creates or checks the arena, and replies
// with the arena's file descriptor attached.
func (s *Server) hello(ss *session) error {
	if err := ss.conn.SetReadDeadline(time.Now().Add(helloTimeout)); err != nil {
		return err
	}
	req, err := protocol.Read(ss.conn)
	if err != nil {
		return err
	}
	if err := ss.conn.SetReadDeadline(time.Time{}); err != nil {
		return err
	}
	resp := &protocol.Message{ID: req.ID, Op: protocol.OpHello}
	var arena *pool.Arena
	switch {
	case req.Op != protocol.OpHello:
		resp.Error = "first request must be hello"
	case req.Version != protocol.Version:
		resp.Error = fmt.Sprintf("protocol version %d not supported; daemon speaks %d", req.Version, protocol.Version)
	case req.Namespace == "":
		resp.Error = "hello must name a namespace"
	case req.SlotBytes == 0:
		// Attach to the existing pool without knowing its slot size. An
		// engine-side component that does not see block sizes (a TPU
		// scheduler) uses this after its worker has created the pool.
		if arena = s.store.Arena(); arena == nil {
			resp.Error = "no pool yet: a client with a slot size must connect first"
		}
	default:
		arena, err = s.store.EnsureArena(req.SlotBytes)
		if err != nil {
			resp.Error = err.Error()
		}
	}
	if resp.Error != "" {
		frame, err := protocol.Encode(resp)
		if err != nil {
			s.log.Error("encode failed", "session", ss.id, "err", err)
		} else if _, err := ss.conn.Write(frame); err != nil {
			s.log.Debug("hello reply failed", "session", ss.id, "err", err)
		}
		return errors.New(resp.Error)
	}
	ss.ns = req.Namespace
	ss.client = req.Client
	resp.Session = uint64(ss.id)
	resp.ArenaBytes = arena.Size()
	resp.NumSlots = arena.NumSlots()
	resp.SlotBytes = arena.SlotBytes()
	frame, err := protocol.Encode(resp)
	if err != nil {
		return err
	}
	_, _, err = ss.conn.WriteMsgUnix(frame, unix.UnixRights(arena.FD()), nil)
	return err
}

// dispatchAndReply runs req and queues its reply. Requests that change the
// pool, or read it whole, run under s.order together with their events and
// reply. So a session sees its own changes, other sessions' changes and its
// snapshots in the order they happened, and never applies a stale event
// over a newer one.
func (s *Server) dispatchAndReply(ss *session, req *protocol.Message) bool {
	switch req.Op {
	case protocol.OpAlloc, protocol.OpCommit, protocol.OpSnapshot:
		s.order.Lock()
		defer s.order.Unlock()
	}
	return s.reply(ss, s.dispatch(ss, req))
}

func (s *Server) dispatch(ss *session, req *protocol.Message) *protocol.Message {
	resp := &protocol.Message{ID: req.ID, Op: req.Op}
	switch req.Op {
	case protocol.OpAlloc:
		slots, status, evicted, err := s.store.Alloc(ss.id, ss.ns, req.Keys)
		if err != nil {
			resp.Error = err.Error()
			break
		}
		resp.Slots = slots
		resp.Status = make([]uint8, len(status))
		for i, st := range status {
			resp.Status[i] = uint8(st)
		}
		// Every session, the caller included, gets an event, so engines
		// drop the keys from their mirrors in order. The response lists
		// them too, for callers that do not track events.
		resp.Removed = keysIn(evicted, ss.ns)
		s.broadcastRemoved(evicted)
	case protocol.OpCommit:
		stored, err := s.store.Commit(ss.id, ss.ns, req.Keys, req.OK)
		if err != nil {
			resp.Error = err.Error()
			break
		}
		s.broadcastStored(ss.ns, stored)
	case protocol.OpPin:
		resp.Slots = s.store.Pin(ss.id, ss.ns, req.Keys)
	case protocol.OpUnpin:
		s.store.Unpin(ss.id, ss.ns, req.Keys)
	case protocol.OpSnapshot:
		resp.Keys = s.store.Snapshot(ss.ns)
	case protocol.OpStats:
		st := s.store.Stats()
		resp.Stats = map[string]any{
			"slot_bytes": st.SlotBytes, "num_slots": st.NumSlots,
			"used_slots": st.UsedSlots, "ready_slots": st.ReadySlots,
			"pinned_slots": st.PinnedSlots, "stores": st.Stores,
			"store_no_space": st.StoreNoSpace, "loads": st.Loads,
			"load_misses": st.LoadMisses, "evictions": st.Evictions,
		}
	default:
		resp.Error = fmt.Sprintf("unknown op %q", req.Op)
	}
	return resp
}

func keysIn(refs []pool.KeyRef, ns string) [][]byte {
	var keys [][]byte
	for _, r := range refs {
		if r.Namespace == ns {
			keys = append(keys, r.Key)
		}
	}
	return keys
}

// broadcastStored tells every session in ns that keys became loadable.
// Callers hold s.order.
func (s *Server) broadcastStored(ns string, refs []pool.KeyRef) {
	if len(refs) == 0 {
		return
	}
	frame, err := protocol.Encode(&protocol.Message{Op: protocol.OpEvent, Stored: keysIn(refs, ns)})
	if err != nil {
		s.log.Error("encode event", "err", err)
		return
	}
	for _, ss := range s.sessionsIn(ns) {
		ss.send(frame)
	}
}

// broadcastRemoved tells every session which of its namespace's keys were
// removed. Callers hold s.order.
func (s *Server) broadcastRemoved(refs []pool.KeyRef) {
	if len(refs) == 0 {
		return
	}
	byNS := make(map[string][][]byte)
	for _, r := range refs {
		byNS[r.Namespace] = append(byNS[r.Namespace], r.Key)
	}
	for ns, keys := range byNS {
		frame, err := protocol.Encode(&protocol.Message{Op: protocol.OpEvent, Removed: keys})
		if err != nil {
			s.log.Error("encode event", "err", err)
			continue
		}
		for _, ss := range s.sessionsIn(ns) {
			ss.send(frame)
		}
	}
}

func (s *Server) sessionsIn(ns string) []*session {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []*session
	for _, ss := range s.sessions {
		if ss.ns == ns {
			out = append(out, ss)
		}
	}
	return out
}

// NumSessions returns the number of connected clients.
func (s *Server) NumSessions() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.sessions)
}
