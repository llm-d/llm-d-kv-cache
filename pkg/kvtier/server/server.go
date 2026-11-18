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

	"golang.org/x/sys/unix"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvtier/pool"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvtier/protocol"
)

// outQueue bounds the frames waiting for a slow client. A client that falls
// this far behind is disconnected; it reconnects and resynchronises from a
// snapshot.
const outQueue = 4096

// Server accepts shim connections and applies their requests to a Store.
type Server struct {
	store *pool.Store
	log   *slog.Logger

	nextID   atomic.Uint64
	mu       sync.Mutex
	sessions map[pool.SessionID]*session
	ln       *net.UnixListener
	wg       sync.WaitGroup
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
	return &Server{store: store, log: log, sessions: make(map[pool.SessionID]*session)}
}

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
		s.wg.Add(1)
		go func() {
			defer s.wg.Done()
			s.handle(conn)
		}()
	}
}

// Close stops accepting, disconnects every session and waits for them.
func (s *Server) Close() error {
	var err error
	if s.ln != nil {
		err = s.ln.Close()
	}
	s.mu.Lock()
	for _, ss := range s.sessions {
		ss.close()
	}
	s.mu.Unlock()
	s.wg.Wait()
	return err
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
		s.store.ReleaseSession(ss.id)
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
		if !s.reply(ss, s.dispatch(ss, req)) {
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
	req, err := protocol.Read(ss.conn)
	if err != nil {
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
		// The caller learns of its own namespace's evictions from the
		// response. Every other session gets an event, so other engines
		// drop the keys from their mirrors.
		resp.Removed = keysIn(evicted, ss.ns)
		s.broadcastRemoved(evicted, ss.id)
	case protocol.OpCommit:
		stored, err := s.store.Commit(ss.id, ss.ns, req.Keys, req.OK)
		if err != nil {
			resp.Error = err.Error()
			break
		}
		s.broadcastStored(ss.ns, stored, ss.id)
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

// broadcastStored tells every other session in ns that keys became loadable.
func (s *Server) broadcastStored(ns string, refs []pool.KeyRef, origin pool.SessionID) {
	if len(refs) == 0 {
		return
	}
	frame, err := protocol.Encode(&protocol.Message{Op: protocol.OpEvent, Stored: keysIn(refs, ns)})
	if err != nil {
		s.log.Error("encode event", "err", err)
		return
	}
	for _, ss := range s.sessionsIn(ns) {
		if ss.id != origin {
			ss.send(frame)
		}
	}
}

// broadcastRemoved tells every session, except origin, which of its
// namespace's keys were evicted.
func (s *Server) broadcastRemoved(refs []pool.KeyRef, origin pool.SessionID) {
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
			if ss.id != origin {
				ss.send(frame)
			}
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
