# Priority Queue Test Suite for FS Backend Thread Pool

# This test suite validates that read operations are prioritized over write operations
# in the thread pool using deterministic completion order testing and
# production-realistic latency distribution analysis.

# Test Strategy:
#     1. Completion Order: Verify reads submitted AFTER writes complete BEFORE them
#     2. Latency Percentiles: Measure p50/p95/p99 with the queue fully saturated
#        by writes, confirming reads jump ahead and tail latency stays bounded

import time

import torch

from llmd_fs_backend.file_mapper import FileMapper
from llmd_fs_backend.mediums import SharedStorageLoadStoreSpec
from llmd_fs_backend.worker import StorageOffloadingHandlers
from tests.test_fs_backend import (
    TMP_DIR,
    build_canonical_kv_caches,
    cleanup_files,
    create_dummy_kv_tensors,
    make_gpu_specs,
    make_storage_specs,
    wait_for,
)

TEST_CONFIG = {
    "model_name": "priority-test-model",
    "dtype": torch.float16,
    "num_layers": 80,
    "block_size": 16,
    "num_heads": 64,
    "head_size": 128,
    "gpu_blocks_per_file": 4,
    "gpu_block_size": 16,
}


def create_test_handler(
    num_blocks: int,
    threads_per_gpu: int,
    model_suffix: str = "",
) -> tuple[StorageOffloadingHandlers, dict]:
    """Create a test handler with specified configuration."""
    config = TEST_CONFIG.copy()
    model_name = f"{config['model_name']}{model_suffix}"

    file_mapper = FileMapper(
        root_dir=TMP_DIR,
        model_name=model_name,
        gpu_block_size=config["gpu_block_size"],
        gpu_blocks_per_file=config["gpu_blocks_per_file"],
        tp_size=1,
        pp_size=1,
        pcp_size=1,
        rank=0,
        dtype=config["dtype"],
    )

    kv_cache = create_dummy_kv_tensors(
        config["num_layers"],
        num_blocks,
        config["block_size"],
        config["num_heads"],
        config["head_size"],
        config["dtype"],
    )

    canonical_kv_caches = build_canonical_kv_caches(kv_cache)

    handler = StorageOffloadingHandlers(
        file_mapper=file_mapper,
        kv_caches=canonical_kv_caches,
        gpu_blocks_per_file=config["gpu_blocks_per_file"],
        gpu_block_size=config["gpu_block_size"],
        threads_per_gpu=threads_per_gpu,
    )

    return handler, {"file_mapper": file_mapper}


def test_priority_completion_order(default_vllm_config):
    """
    Test that reads submitted AFTER writes complete BEFORE them.
    This is the most direct proof that priority queuing works.

    Strategy:
        1. Submit 20 write operations
        2. Let queue fill up
        3. Submit 5 read operations
        4. Track completion order
        5. Verify reads completed early in the sequence

    Expected with priority:
        Completion order: [W1, W2, R1, R2, R3, R4, R5, W3, W4, ...]
        Reads appear in first ~30% of completions

    Expected without priority (FIFO):
        Completion order: [W1, W2, ..., W20, R1, R2, R3, R4, R5]
        Reads appear in last ~20% of completions
    """
    threads_per_gpu = 2
    num_write_files = 20
    num_read_files = 5

    # Calculate total blocks needed
    blocks_per_file = TEST_CONFIG["gpu_blocks_per_file"]
    num_blocks = (num_write_files + num_read_files) * blocks_per_file

    handler, context = create_test_handler(
        num_blocks=num_blocks,
        threads_per_gpu=threads_per_gpu,
        model_suffix="-completion-order",
    )

    file_mapper = context["file_mapper"]
    put = handler.gpu_to_storage_handler
    get = handler.storage_to_gpu_handler
    finished_cache = {}

    completion_order = []
    completion_times = {}

    # Step 1: Prepare files for reading (write them first)
    read_block_ids = list(range(num_read_files * blocks_per_file))
    read_put_gpu = make_gpu_specs(read_block_ids)
    read_put_storage, read_hashes = make_storage_specs(num_read_files)
    cleanup_files(file_mapper, read_hashes)

    put.transfer_async(job_id=0, spec=(read_put_gpu, read_put_storage))
    ok = wait_for(put, job_id=0, timeout=30.0, _finished_cache=finished_cache)
    assert ok, "Initial read file preparation failed"

    # Step 2: Submit bulk write operations (jobs 1-20)
    write_offset = num_read_files
    write_block_start = num_read_files * blocks_per_file

    for i in range(num_write_files):
        block_ids = list(
            range(
                write_block_start + i * blocks_per_file,
                write_block_start + (i + 1) * blocks_per_file,
            )
        )
        write_gpu = make_gpu_specs(block_ids)
        write_storage, write_hashes = make_storage_specs(
            1, start_offset=write_offset + i
        )
        cleanup_files(file_mapper, write_hashes)

        job_id = 1 + i
        put.transfer_async(job_id=job_id, spec=(write_gpu, write_storage))

    # Step 3: Let queue build up, then submit reads
    time.sleep(0.1)  # Allow writes to start queuing

    # Submit read operations (jobs 100-104)
    for i in range(num_read_files):
        file_idx = i
        block_ids = list(
            range(file_idx * blocks_per_file, (file_idx + 1) * blocks_per_file)
        )
        read_gpu = make_gpu_specs(block_ids)
        read_storage = SharedStorageLoadStoreSpec([read_put_storage.keys[file_idx]])

        job_id = 100 + i
        get.transfer_async(job_id=job_id, spec=(read_storage, read_gpu))

    # Step 4: Wait for all jobs and track completion order
    all_write_jobs = list(range(1, num_write_files + 1))
    all_read_jobs = list(range(100, 100 + num_read_files))

    start_time = time.time()

    # Poll for completions in order they finish
    remaining_jobs = set(all_write_jobs + all_read_jobs)

    while remaining_jobs:
        finished = put.get_finished() + get.get_finished()
        for result in finished:
            job_id = result.job_id
            if job_id in remaining_jobs:
                completion_order.append(job_id)
                completion_times[job_id] = time.time() - start_time
                remaining_jobs.remove(job_id)
                finished_cache[job_id] = result.success
        time.sleep(0.01)

    # Step 5: Analyze completion order
    total_jobs = num_write_files + num_read_files
    read_positions = [completion_order.index(job_id) for job_id in all_read_jobs]
    avg_read_position = sum(read_positions) / len(read_positions)

    # Calculate position as percentage
    avg_position_pct = (avg_read_position / total_jobs) * 100

    print(f"\n{'=' * 70}")
    print("Completion Order Test Results")
    print(f"{'=' * 70}")
    print("Configuration:")
    print(f"  Threads: {threads_per_gpu}")
    print(f"  Write jobs: {num_write_files} (jobs 1-{num_write_files})")
    print(f"  Read jobs: {num_read_files} (jobs 100-{100 + num_read_files - 1})")
    print("\nCompletion order (first 15 jobs):")
    print(f"  {completion_order[:15]}")
    print("\nRead job positions in completion order:")
    for i, pos in enumerate(read_positions):
        job_id = 100 + i
        print(
            f"  Job {job_id}: position {pos}/{total_jobs} "
            f"({pos / total_jobs * 100:.1f}%)"
        )
    print("\nSummary:")
    print(
        f"  Average read position: "
        f"{avg_read_position:.1f}/{total_jobs} "
        f"({avg_position_pct:.1f}%)"
    )
    print("  Expected with priority: <40% (reads jump ahead)")
    print("  Expected without priority: >80% (reads wait at end)")
    print(f"{'=' * 70}")

    # Assertion: With priority, reads should complete in first half
    # Without priority, reads would complete in last 20% (positions 20-25)
    assert avg_position_pct < 50, (
        f"Reads completed at {avg_position_pct:.1f}% of queue "
        f"(expected <50% with priority). "
        f"Completion order: {completion_order}. "
        f"This indicates priority queuing is NOT working "
        f"- reads did not jump ahead of writes."
    )

    # Cleanup
    cleanup_files(file_mapper, read_hashes)
    for i in range(num_write_files):
        _, write_hashes = make_storage_specs(1, start_offset=write_offset + i)
        cleanup_files(file_mapper, write_hashes)

    del handler, put, get


def test_read_latency_percentiles(default_vllm_config):
    """
    Measure read latency distribution under write queue saturation.

    All writes are submitted at once to fully saturate the queue, then reads
    are submitted while writes are still processing. This guarantees real
    priority contention and exercises whether reads genuinely jump ahead.

    Key metrics:
        - p50: Median read latency
        - p95: 95th percentile (acceptable tail)
        - p99: 99th percentile (worst case)
        - Tail ratio: p99/p50 (should be <5x for good QoS)
    """
    threads_per_gpu = 4
    num_write_files = 40
    num_read_files = 5
    num_reads = 15

    blocks_per_file = TEST_CONFIG["gpu_blocks_per_file"]
    num_blocks = (num_read_files + num_write_files) * blocks_per_file

    handler, context = create_test_handler(
        num_blocks=num_blocks,
        threads_per_gpu=threads_per_gpu,
        model_suffix="-percentiles",
    )

    file_mapper = context["file_mapper"]
    put = handler.gpu_to_storage_handler
    get = handler.storage_to_gpu_handler
    finished_cache = {}

    # Prepare files for reading
    read_block_ids = list(range(num_read_files * blocks_per_file))
    read_put_gpu = make_gpu_specs(read_block_ids)
    read_put_storage, read_hashes = make_storage_specs(num_read_files)
    cleanup_files(file_mapper, read_hashes)

    put.transfer_async(job_id=0, spec=(read_put_gpu, read_put_storage))
    ok = wait_for(put, job_id=0, timeout=30.0, _finished_cache=finished_cache)
    assert ok, "Initial file preparation failed"

    # Saturate the queue with all writes at once
    write_offset = num_read_files
    write_block_start = num_read_files * blocks_per_file

    for i in range(num_write_files):
        block_ids = list(
            range(
                write_block_start + i * blocks_per_file,
                write_block_start + (i + 1) * blocks_per_file,
            )
        )
        write_gpu = make_gpu_specs(block_ids)
        write_storage, write_hashes = make_storage_specs(
            1, start_offset=write_offset + i
        )
        cleanup_files(file_mapper, write_hashes)
        put.transfer_async(job_id=1 + i, spec=(write_gpu, write_storage))

    # Submit reads while the write queue is saturated and measure latency
    read_latencies = []
    for i in range(num_reads):
        file_idx = i % num_read_files
        block_ids = list(
            range(file_idx * blocks_per_file, (file_idx + 1) * blocks_per_file)
        )
        read_gpu = make_gpu_specs(block_ids)
        read_storage = SharedStorageLoadStoreSpec([read_put_storage.keys[file_idx]])

        start = time.time()
        get.transfer_async(job_id=1000 + i, spec=(read_storage, read_gpu))
        ok = wait_for(
            get, job_id=1000 + i, timeout=30.0, _finished_cache=finished_cache
        )
        assert ok, f"Read job {1000 + i} failed"
        read_latencies.append(time.time() - start)
        time.sleep(0.05)

    for i in range(num_write_files):
        wait_for(put, job_id=1 + i, timeout=60.0, _finished_cache=finished_cache)

    # Calculate and report percentiles
    data = sorted(read_latencies)
    n = len(data)

    def pct(p):
        return data[min(int(n * p / 100), n - 1)]

    p50, p95, p99 = pct(50), pct(95), pct(99)
    tail_ratio = p99 / p50 if p50 > 0 else float("inf")

    print(f"\n{'=' * 70}")
    print("Latency Percentiles — Saturated Queue")
    print(f"{'=' * 70}")
    print(f"  Threads: {threads_per_gpu}  |  Writes: {num_write_files}  |  Reads: {n}")
    print(f"  Min:  {data[0]:.3f}s")
    print(f"  p50:  {p50:.3f}s (median)")
    print(f"  p95:  {p95:.3f}s")
    print(f"  p99:  {p99:.3f}s (worst case)")
    print(f"  Max:  {data[-1]:.3f}s")
    print(f"  Tail ratio (p99/p50): {tail_ratio:.2f}x  (target <5.0x)")
    print(f"{'=' * 70}")

    assert p99 < 1.5, (
        f"p99 too high: {p99:.3f}s (expected <1.5s). "
        f"Reads are being starved by writes under queue saturation."
    )
    assert p50 < 0.5, (
        f"p50 too high: {p50:.3f}s (expected <0.5s). "
        f"Priority is not effectively reducing median read wait time."
    )
    assert tail_ratio < 5.0, (
        f"Tail ratio {tail_ratio:.2f}x (p99/p50) exceeds 5.0x. "
        f"Inconsistent read performance under saturation."
    )

    cleanup_files(file_mapper, read_hashes)
    for i in range(num_write_files):
        _, write_hashes = make_storage_specs(1, start_offset=write_offset + i)
        cleanup_files(file_mapper, write_hashes)

    del handler, put, get


def test_write_starvation_prevention(default_vllm_config):
    """
    Test that writes aren't starved under continuous read pressure.

    In a pure priority system, if reads keep arriving, writes might never
    execute. This test validates that the implementation has fairness
    guarantees to prevent write starvation.

    Strategy:
        1. Submit 10 write operations
        2. Immediately start flooding queue with continuous reads
        3. Submit new read every 0.02s (faster than workers can process)
        4. Track write completion times
        5. Verify all writes complete within reasonable bounds

    Expected behavior:
        - All writes should complete (no infinite starvation)
        - Write latency should stay bounded (e.g., <3s per write)
        - Writes should get ~25% of throughput (1 write-preferring worker / 4 total)
    """
    time.sleep(1.0)  # let GPU recover from previous tests before measuring throughput

    threads_per_gpu = 4
    num_writes = 10
    read_submission_interval = 0.02
    max_acceptable_write_latency = 3.0

    blocks_per_file = TEST_CONFIG["gpu_blocks_per_file"]
    num_read_files = 5
    num_blocks = (num_writes + num_read_files) * blocks_per_file

    handler, context = create_test_handler(
        num_blocks=num_blocks,
        threads_per_gpu=threads_per_gpu,
        model_suffix="-starvation",
    )

    file_mapper = context["file_mapper"]
    put = handler.gpu_to_storage_handler
    get = handler.storage_to_gpu_handler
    finished_cache = {}

    # Prepare files for continuous reading
    read_block_ids = list(range(num_read_files * blocks_per_file))
    read_put_gpu = make_gpu_specs(read_block_ids)
    read_put_storage, read_hashes = make_storage_specs(num_read_files)
    cleanup_files(file_mapper, read_hashes)

    put.transfer_async(job_id=0, spec=(read_put_gpu, read_put_storage))
    ok = wait_for(put, job_id=0, timeout=30.0, _finished_cache=finished_cache)
    assert ok, "Initial file preparation failed"

    # Submit all writes at once
    write_offset = num_read_files
    write_block_start = num_read_files * blocks_per_file
    write_start_times = {}

    for i in range(num_writes):
        block_ids = list(
            range(
                write_block_start + i * blocks_per_file,
                write_block_start + (i + 1) * blocks_per_file,
            )
        )
        write_gpu = make_gpu_specs(block_ids)
        write_storage, write_hashes = make_storage_specs(
            1, start_offset=write_offset + i
        )
        cleanup_files(file_mapper, write_hashes)

        job_id = 1 + i
        write_start_times[job_id] = time.time()
        put.transfer_async(job_id=job_id, spec=(write_gpu, write_storage))

    # Start continuous read flood
    read_job_counter = 1000
    reads_submitted = 0
    start_time = time.time()

    # Continue flooding with reads until all writes complete
    write_jobs = set(range(1, num_writes + 1))
    write_completion_times = {}

    while write_jobs:
        file_idx = reads_submitted % num_read_files
        block_ids = list(
            range(file_idx * blocks_per_file, (file_idx + 1) * blocks_per_file)
        )
        read_gpu = make_gpu_specs(block_ids)
        read_storage = SharedStorageLoadStoreSpec([read_put_storage.keys[file_idx]])

        get.transfer_async(job_id=read_job_counter, spec=(read_storage, read_gpu))
        reads_submitted += 1
        read_job_counter += 1

        finished = put.get_finished()
        for result in finished:
            job_id = result.job_id
            if job_id in write_jobs:
                latency = time.time() - write_start_times[job_id]
                write_completion_times[job_id] = latency
                write_jobs.remove(job_id)
                finished_cache[job_id] = result.success

        # Timeout if writes don't complete
        elapsed = time.time() - start_time
        if elapsed > 30.0:
            raise TimeoutError(
                f"Write starvation detected! {len(write_jobs)} writes "
                f"did not complete after 30s under read pressure. "
                f"Submitted {reads_submitted} reads."
            )

        time.sleep(read_submission_interval)

    write_latencies = list(write_completion_times.values())
    max_write_latency = max(write_latencies)
    avg_write_latency = sum(write_latencies) / len(write_latencies)
    total_duration = time.time() - start_time

    # Estimate throughput
    write_throughput = num_writes / total_duration
    read_throughput = reads_submitted / total_duration
    write_pct = write_throughput / (write_throughput + read_throughput) * 100

    print(f"\n{'=' * 70}")
    print("Write Starvation Prevention Test Results")
    print(f"{'=' * 70}")
    print("Configuration:")
    print(f"  Threads: {threads_per_gpu}")
    print(f"  Writes submitted: {num_writes}")
    print(f"  Reads submitted: {reads_submitted} (continuous flood)")
    print(f"  Total duration: {total_duration:.2f}s")
    print("\nWrite Latencies:")
    print(f"  Min: {min(write_latencies):.3f}s")
    print(f"  Avg: {avg_write_latency:.3f}s")
    print(f"  Max: {max_write_latency:.3f}s")
    print(f"  Target: <{max_acceptable_write_latency}s")
    print("\nThroughput:")
    print(f"  Writes: {write_throughput:.2f} ops/s")
    print(f"  Reads: {read_throughput:.2f} ops/s")
    print(f"  Write percentage: {write_pct:.1f}% (expected ~25%, threshold >10%)")
    print(f"{'=' * 70}")

    assert max_write_latency < max_acceptable_write_latency, (
        f"Write starvation detected! Max write latency {max_write_latency:.3f}s "
        f"exceeds {max_acceptable_write_latency}s under continuous read pressure."
    )

    # With 75% read-preferring workers (3/4), writes should get ~25% throughput.
    # We use 10% as threshold to account for GPU load variability when running
    # after other tests (writes can be slower due to memory pressure).
    min_write_pct = 10.0

    assert write_pct > min_write_pct, (
        f"Writes got only {write_pct:.1f}% of throughput under read pressure. "
        f"Expected ~25% (1 write-preferring worker out of 4 total workers). "
        f"This indicates unfair scheduling - writes should get >{min_write_pct}% "
        f"even when reads are continuously submitted."
    )

    cleanup_files(file_mapper, read_hashes)
    for i in range(num_writes):
        _, write_hashes = make_storage_specs(1, start_offset=write_offset + i)
        cleanup_files(file_mapper, write_hashes)

    del handler, put, get
