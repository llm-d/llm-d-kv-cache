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

// Command kvtierd is the llm-d KV tiering daemon. It runs one per node,
// owns a shared-memory pool of KV blocks, and serves engine-side shims over
// a Unix socket.
package main

import (
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvtier/policy"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvtier/pool"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvtier/server"
)

func main() {
	var (
		socket      = flag.String("socket", "/var/run/llmd-kv-tiering/kvtierd.sock", "Unix socket path")
		capacity    = flag.String("capacity", "16GiB", "pool capacity, e.g. 512GiB")
		slotBytes   = flag.Uint64("slot-bytes", 0, "block size in bytes; 0 takes it from the first client")
		hugetlb     = flag.Bool("hugetlb", false, "back the pool with 2 MiB huge pages")
		prefault    = flag.Bool("prefault", true, "touch the pool at creation, so its memory is charged to this container")
		policyName  = flag.String("policy", "lru", "eviction policy")
		metricsAddr = flag.String("metrics-addr", ":9400", "address for /metrics and /healthz; empty disables")
		logLevel    = flag.String("log-level", "info", "debug, info, warn or error")
	)
	flag.Parse()

	var lvl slog.Level
	if err := lvl.UnmarshalText([]byte(*logLevel)); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: lvl}))

	capBytes, err := parseBytes(*capacity)
	if err != nil {
		log.Error("bad --capacity", "err", err)
		os.Exit(2)
	}
	if policy.New(*policyName) == nil {
		log.Error("unknown --policy", "policy", *policyName)
		os.Exit(2)
	}
	name := *policyName
	store := pool.NewStore(pool.ArenaOptions{
		CapacityBytes: capBytes,
		SlotBytes:     *slotBytes,
		HugeTLB:       *hugetlb,
		Prefault:      *prefault,
	}, func() policy.Policy { return policy.New(name) })
	if *slotBytes != 0 {
		if _, err := store.EnsureArena(*slotBytes); err != nil {
			log.Error("create pool", "err", err)
			os.Exit(1)
		}
	}

	srv := server.New(store, log)
	if err := srv.Listen(*socket); err != nil {
		log.Error("listen", "socket", *socket, "err", err)
		os.Exit(1)
	}
	log.Info("kvtierd listening", "socket", *socket, "capacity", capBytes, "policy", name)

	if *metricsAddr != "" {
		mux := http.NewServeMux()
		mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
			if _, err := w.Write([]byte("ok\n")); err != nil {
				log.Debug("healthz write", "err", err)
			}
		})
		mux.HandleFunc("/metrics", func(w http.ResponseWriter, _ *http.Request) { writeMetrics(w, store, srv) })
		go func() {
			metricsServer := &http.Server{Addr: *metricsAddr, Handler: mux, ReadHeaderTimeout: 10 * time.Second}
			if err := metricsServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
				log.Error("metrics server", "err", err)
			}
		}()
	}

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sig
		log.Info("shutting down")
		srv.Close()
	}()
	if err := srv.Serve(); err != nil {
		log.Error("serve", "err", err)
		os.Exit(1)
	}
	os.Remove(*socket)
}

func writeMetrics(w http.ResponseWriter, store *pool.Store, srv *server.Server) {
	st := store.Stats()
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	gauge := func(name, help string, v any) {
		fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s gauge\n%s %v\n", name, help, name, name, v)
	}
	counter := func(name, help string, v any) {
		fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s counter\n%s %v\n", name, help, name, name, v)
	}
	gauge("kvtierd_slot_bytes", "Size of one pool slot.", st.SlotBytes)
	gauge("kvtierd_slots", "Slots in the pool.", st.NumSlots)
	gauge("kvtierd_slots_used", "Slots holding or receiving a block.", st.UsedSlots)
	gauge("kvtierd_slots_ready", "Slots holding a loadable block.", st.ReadySlots)
	gauge("kvtierd_slots_pinned", "Slots pinned by a load in progress.", st.PinnedSlots)
	gauge("kvtierd_clients", "Connected engine-side clients.", srv.NumSessions())
	counter("kvtierd_stores_total", "Blocks stored.", st.Stores)
	counter("kvtierd_store_no_space_total",
		"Blocks not stored because the pool was full of pinned or in-flight blocks.", st.StoreNoSpace)
	counter("kvtierd_loads_total", "Blocks pinned for a load.", st.Loads)
	counter("kvtierd_load_misses_total", "Blocks requested for a load but absent.", st.LoadMisses)
	counter("kvtierd_evictions_total", "Blocks evicted.", st.Evictions)
}

// parseBytes parses sizes such as 512GiB, 64G, 1048576.
func parseBytes(text string) (uint64, error) {
	text = strings.TrimSpace(text)
	units := []struct {
		suffix string
		mult   uint64
	}{
		{"TiB", 1 << 40},
		{"GiB", 1 << 30},
		{"MiB", 1 << 20},
		{"KiB", 1 << 10},
		{"TB", 1e12},
		{"GB", 1e9},
		{"MB", 1e6},
		{"KB", 1e3},
		{"T", 1 << 40},
		{"G", 1 << 30},
		{"M", 1 << 20},
		{"K", 1 << 10},
	}
	for _, u := range units {
		if strings.HasSuffix(text, u.suffix) {
			n, err := strconv.ParseFloat(strings.TrimSuffix(text, u.suffix), 64)
			if err != nil {
				return 0, err
			}
			return uint64(n * float64(u.mult)), nil
		}
	}
	return strconv.ParseUint(text, 10, 64)
}
