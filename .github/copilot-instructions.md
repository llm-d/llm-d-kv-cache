# Copilot Cloud Agent Instructions

## Repository Overview

This is **llm-d-kv-cache**, a pluggable Go library and service that enables **KV-Cache Aware Routing** for distributed vLLM-based LLM inference. It keeps a near-real-time global view of KV-cache block locality across a fleet of vLLM (and SGLang) pods, exposing fast pod-scoring to inference schedulers.

Module path: `github.com/llm-d/llm-d-kv-cache`  
Go version: extracted at runtime from `go.mod` (currently 1.24.1).  
License: Apache 2.0 — every new `.go` file must carry the standard Apache 2.0 header (see `pkg/kvevents/pool.go` for the exact format).

---

## Repository Layout

```
.
├── api/               # Protobuf definitions & generated gRPC stubs (Go + Python)
│   ├── indexerpb/     # KV-Cache Indexer gRPC API
│   └── tokenizerpb/   # Tokenizer gRPC API
├── benchmarking/      # Benchmark utilities
├── deploy/            # Kubernetes / Helm deployment manifests
├── docs/              # Architecture and configuration documentation
│   ├── architecture.md
│   └── configuration.md
├── examples/          # Runnable reference implementations
├── hack/              # Developer tooling scripts
├── hooks/             # Git hooks
├── kv_connectors/     # External KV-store connectors (Python)
│   ├── llmd_fs_backend/   # Filesystem-backed KV connector
│   └── pvc_evictor/       # Kubernetes PVC evictor
├── pkg/               # Core Go library packages (see below)
├── services/
│   └── uds_tokenizer/ # Python gRPC tokenizer service over Unix Domain Socket
├── tests/             # Integration / e2e test helpers
├── vllm-setup-helm/   # Helm chart helpers for vLLM setup
├── Dockerfile
├── Makefile           # All developer workflows — primary entry point
├── go.mod / go.sum
├── .golangci.yml      # golangci-lint v2 config
└── .pre-commit-config.yaml
```

### Core Go packages under `pkg/`

| Package | Responsibility |
|---|---|
| `pkg/kvcache` | Main `Indexer` orchestrator — scores pods for a given prompt |
| `pkg/kvcache/kvblock` | KV-block index backends (in-memory LRU, Redis, Valkey, cost-aware), token processor, scorer |
| `pkg/kvevents` | Event pool, ZMQ subscriber, `EngineAdapter` interface, raw/generic event types |
| `pkg/kvevents/engineadapter` | vLLM and SGLang adapter implementations |
| `pkg/tokenization` | Tokenizer pool (embedded Python tokenizer or UDS gRPC tokenizer) |
| `pkg/preprocessing` | Chat-completion rendering (Python `chat_completions` via CGo) |
| `pkg/telemetry` | OpenTelemetry tracing helpers |
| `pkg/utils` | Logging and misc utilities |

---

## System Architecture Summary

Two primary data flows:

**Write Path** (event ingestion):  
`vLLM pod` → ZMQ pub/sub → `kvevents.zmqSubscriber` → `kvevents.Pool` (sharded by pod-id via FNV-1a) → `EngineAdapter.ParseMessage()` → `kvblock.Index.Add/Evict()`

**Read Path** (pod scoring):  
`Scheduler` → `kvcache.Indexer.GetPodScores(prompt, model, pods[])` → tokenize → `kvblock.TokenProcessor.TokensToKVBlockKeys()` → `kvblock.Index.Lookup()` → `kvblock.Scorer.Score()` → scores map

Key design points:
- **KV-block hashing** uses FNV-64a over CBOR-encoded `[parentHash, tokenChunk, extra]` tuples, chained from a configurable seed. This seed **must** match `PYTHONHASHSEED` in vLLM pods.
- **Engine adapters** parse engine-specific message formats: vLLM uses msgpack with positional `[]any` arrays (`array_like + omit_defaults`); SGLang uses a different format. Positional decoding is intentional for forward/backward compatibility.
- **Tokenizers** can be either embedded (Python via CGo, requires `CGO_ENABLED=1`) or external via UDS gRPC. The UDS path is the default for CI and is simpler to build.

---

## Build, Lint, and Test

### System Prerequisites

Before any Go build or test, the `libzmq3-dev` system library must be installed:

```bash
sudo apt-get install -y libzmq3-dev pkg-config
```

For lint and embedded-tokenizer builds, also install:

```bash
sudo apt-get install -y python3.12-dev python3.12-venv clang-format
```

### Key Makefile Targets

```bash
make unit-test          # Run unit tests (UDS tokenizer path, no Python needed)
make unit-test-race     # Unit tests with Go race detector
make e2e-test           # End-to-end tests (builds UDS tokenizer Docker image)
make lint               # Run golangci-lint + pre-commit hooks (precommit target)
make precommit          # tidy-go + lint + copr-fix (run before every commit)
make build-uds          # Build without embedded tokenizers
make build-embedded     # Build with embedded Python tokenizers (CGo, Python required)
make generate-grpc      # Regenerate gRPC stubs from protobuf definitions
make download-zmq       # Install ZMQ (auto-detects OS/arch; skips if already installed)
make clang              # Check C/C++ formatting with clang-format
```

The Go version is auto-detected from `go.mod`; no need to hard-code it.

### Running Tests Without Embedded Tokenizers (Recommended for CI)

```bash
sudo apt-get install -y libzmq3-dev pkg-config
go mod download
make unit-test-uds      # alias: make unit-test
```

### Linting

The project uses **golangci-lint v2** (`v2.1.6`) with a strict linter set defined in `.golangci.yml`. The linter requires CGo flags for files using the embedded tokenizer:

```bash
CGO_CFLAGS="$(python3.12-config --cflags)"
CGO_LDFLAGS="$(python3.12-config --ldflags --embed) -ldl -lm"
export CGO_CFLAGS CGO_LDFLAGS CGO_ENABLED=1
make precommit
```

Pre-commit hooks also run `ruff` (Python), `typos`, `clang-format`, and `actionlint`.

---

## Code Conventions

### Go Style

- **Test packages**: External test packages (`package foo_test`) are enforced by the `testpackage` linter — all `_test.go` files must use the `_test` suffix package name (except `export_test.go` helper files).
- **Error wrapping**: Use `fmt.Errorf("context: %w", err)` — the `errorlint` linter enforces correct wrapping.
- **Logging**: Use `sigs.k8s.io/controller-runtime/pkg/log` (logr interface) for structured logging; avoid raw `fmt.Print*`.
- **Comments**: All exported symbols must have godoc comments ending with a period (enforced by `godot`).
- **No naked returns**, **no named returns** (`nonamedreturns` linter).
- **Line length**: max 200 characters (`lll` linter).
- **Import grouping**: stdlib, then external, then internal (`grouper` linter).
- **License header**: Every new `.go` file requires the Apache 2.0 header. Year `2025` or `2026` depending on file creation year. See existing files for the exact format (C-style `//` comments).

### Adding a New Engine Adapter

1. Create `pkg/kvevents/engineadapter/<engine>_adapter.go` implementing `kvevents.EngineAdapter`.
2. Register it in `pkg/kvevents/engineadapter/adapter.go`'s `NewAdapter()` switch.
3. Add `<engine>_adapter_test.go` with package `engineadapter_test`.

### Adding a New Index Backend

1. Implement the `kvblock.Index` interface in `pkg/kvcache/kvblock/`.
2. Add the config struct to `IndexConfig` in `index.go`.
3. Wire it in `NewIndex()`.

---

## Protobuf / gRPC

Protobuf sources live in `api/`. Generated files are committed; re-generate with:

```bash
make generate-grpc-go      # Go stubs → api/indexerpb/, api/tokenizerpb/
make generate-grpc-python  # Python stubs → services/uds_tokenizer/tokenizerpb/
```

Requires `protoc` and `grpc_tools` installed.

---

## Configuration

All configs are JSON-serializable Go structs. The top-level config has three sections:

```json
{
  "indexerConfig": { ... },
  "kvEventsConfig": { ... },
  "tokenProcessorConfig": { ... }
}
```

See `docs/configuration.md` for all options and defaults.

Key env variable: `PYTHONHASHSEED` must match the hash seed configured in `TokenProcessorConfig` for KV-block hashes to be consistent with vLLM pods.

---

## CI Workflows

All workflows are in `.github/workflows/`. Key ones:

| Workflow | Trigger | What it does |
|---|---|---|
| `ci-test.yaml` | PR to `main`/`dev` | Unit tests + e2e tests |
| `ci-lint.yaml` | PR to `main`/`dev` | golangci-lint + pre-commit + clang-format |
| `ci-uds-tokenizer.yaml` | PR | UDS tokenizer Python tests |
| `ci-nightly-race.yaml` | Nightly | Unit tests with race detector |
| `ci-examples.yaml` | PR | Build and verify examples |
| `copilot-setup-steps.yml` | Push/manual | Pre-installs dependencies for Copilot cloud agent |

The `copilot-setup-steps.yml` currently only installs the `gh-aw` extension. The system deps (`libzmq3-dev`, `pkg-config`, Go) should be installed by the agent at task time using the Makefile targets.

---

## Common Pitfalls

1. **ZMQ not installed**: All Go builds and tests require `libzmq3-dev`. Run `make download-zmq` or `sudo apt-get install -y libzmq3-dev pkg-config` first.
2. **Embedded tokenizer builds need Python 3.12**: `make build-embedded` / `make unit-test-embedded` require `python3.12-dev` and `python3.12-venv`. Prefer `make unit-test-uds` for a simpler test cycle.
3. **CGo flags for linting**: The linter must see `CGO_CFLAGS`/`CGO_LDFLAGS` from `python3.12-config`; without them, CGo files won't parse.
4. **Hash seed alignment**: When writing tests that cross-validate block hashes with vLLM output, ensure the hash seed (`TokenProcessorConfig.HashSeed`) matches `PYTHONHASHSEED` in the target vLLM environment.
5. **vLLM msgpack positional arrays**: The `VLLMAdapter` decodes payload as `[]any` positional arrays (not named fields). Length guards handle schema evolution — do not switch to named-field decoding.
6. **Testcontainers for Redis/Valkey**: Tests for the Redis and Valkey index backends use `testcontainers-go` and require Docker or Podman.
