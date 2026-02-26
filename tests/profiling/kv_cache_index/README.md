# KV-Cache Index Profiling

This package contains micro-benchmarks for the `llm-d-kv-cache` indexing strategies. It is designed to measure and compare the latency and memory overhead of different storage backends used for the KV-Cache system.

## Benchmarked Implementations

1.  **In-Memory (`memory`)**: Standard Go map implementation. Purely local, non-persistent, and serves as the baseline for maximum speed.
2.  **Redis (`redis`)**: Remote storage implementation. Tests run against an embedded `miniredis` instance to measure driver serialization and protocol overhead without network jitter.
3.  **CostAware (`cost`)**: Smart tiering logic that calculates storage costs.

## Prerequisites

* **Go 1.22+**: Required for `math/rand/v2`.
* **Dependencies**: Run `go mod tidy` to ensure `miniredis` and other dependencies are installed.

## Running the Benchmarks

### Basic Performance Test (Latency)
Run all benchmarks to see execution time per operation:

```bash
go test -bench=.
```

### Memory Statistics
```
go test -bench=. -benchmem
```

### Running specific test

use the -bench option to filter

```
go test -bench=Redis -benchmem
```

### Understanding the Output

`BenchmarkInMemory_Add-12      192   6086106 ns/op    500 B/op      5 allocs/op`

192: Loop iterations (sample size).

6086106 ns/op: Time per operation (~6ms).

500 B/op: Bytes of memory allocated per operation (only visible with -benchmem).

5 allocs/op: Distinct memory allocations per operation (lower is better to reduce GC pressure).


### Visualize
CPU usage

`go tool pprof -http=:8080 cpu.out`
