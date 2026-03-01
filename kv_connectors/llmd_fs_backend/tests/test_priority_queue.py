# Priority Queue Test Suite for FS Backend Thread Pool

# This test suite validates that read operations are prioritized over write operations
# in the thread pool using deterministic completion order testing and
# production-realistic latency distribution analysis.

# Test Strategy:
#     1. Completion Order: Verify reads submitted AFTER writes complete BEFORE them
#     2. Latency Percentiles: Measure p50/p95/p99 under mixed load
#     3. Queue Behavior: Track how reads behave when queue is saturated with writes

import time

import torch
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

from llmd_fs_backend.file_mapper import FileMapper
from llmd_fs_backend.mediums import SharedStorageLoadStoreSpec
from llmd_fs_backend.worker import StorageOffloadingHandlers
from tests.test_fs_backend import (
    TMP_DIR,
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

    attn_backends = {
        f"layer_{i}": FlashAttentionBackend for i in range(config["num_layers"])
    }
    kv_dict = {f"layer_{i}": kv_cache[i] for i in range(config["num_layers"])}

    handler = StorageOffloadingHandlers(
        file_mapper=file_mapper,
        kv_caches=kv_dict,
        gpu_blocks_per_file=config["gpu_blocks_per_file"],
        gpu_block_size=config["gpu_block_size"],
        threads_per_gpu=threads_per_gpu,
        attn_backends=attn_backends,
    )

    return handler, {"file_mapper": file_mapper, "kv_dict": kv_dict}


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
        read_storage = SharedStorageLoadStoreSpec(
            [read_put_storage.block_hashes[file_idx]]
        )

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
        for job_id, ok in finished:
            if job_id in remaining_jobs:
                completion_order.append(job_id)
                completion_times[job_id] = time.time() - start_time
                remaining_jobs.remove(job_id)
                finished_cache[job_id] = ok
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
    Measure read latency distribution under write pressure.

    This test is production-realistic: we care about p50, p95, p99 latency.
    Priority should reduce tail latency (p95, p99) for reads.

    Strategy:
        1. Continuously submit writes (simulating cache evictions)
        2. Intersperse reads (simulating cache hits)
        3. Measure read latency distribution
        4. Verify p99 is reasonable

    Key metrics:
        - p50: Median read latency
        - p95: 95th percentile (acceptable tail)
        - p99: 99th percentile (worst case)
        - Tail ratio: p99/p50 (should be <3x for good QoS)
    """
    threads_per_gpu = 4
    num_write_batches = 10
    num_reads = 20
    writes_per_batch = 3

    # Calculate blocks needed
    blocks_per_file = TEST_CONFIG["gpu_blocks_per_file"]
    num_read_files = 5
    total_write_files = num_write_batches * writes_per_batch
    num_blocks = (num_read_files + total_write_files) * blocks_per_file

    handler, context = create_test_handler(
        num_blocks=num_blocks,
        threads_per_gpu=threads_per_gpu,
        model_suffix="-percentiles",
    )

    file_mapper = context["file_mapper"]
    put = handler.gpu_to_storage_handler
    get = handler.storage_to_gpu_handler
    finished_cache = {}

    # Step 1: Prepare files for reading
    read_block_ids = list(range(num_read_files * blocks_per_file))
    read_put_gpu = make_gpu_specs(read_block_ids)
    read_put_storage, read_hashes = make_storage_specs(num_read_files)
    cleanup_files(file_mapper, read_hashes)

    put.transfer_async(job_id=0, spec=(read_put_gpu, read_put_storage))
    ok = wait_for(put, job_id=0, timeout=30.0, _finished_cache=finished_cache)
    assert ok, "Initial file preparation failed"

    # Step 2: Continuously submit writes and intersperse reads
    read_latencies = []
    write_job_id = 1
    write_file_idx = 0
    write_offset = num_read_files
    write_block_start = num_read_files * blocks_per_file

    for batch_idx in range(num_write_batches):
        # Submit a batch of writes
        for i in range(writes_per_batch):
            block_ids = list(
                range(
                    write_block_start + write_file_idx * blocks_per_file,
                    write_block_start + (write_file_idx + 1) * blocks_per_file,
                )
            )
            write_gpu = make_gpu_specs(block_ids)
            write_storage, write_hashes = make_storage_specs(
                1, start_offset=write_offset + write_file_idx
            )
            cleanup_files(file_mapper, write_hashes)

            put.transfer_async(job_id=write_job_id, spec=(write_gpu, write_storage))
            write_job_id += 1
            write_file_idx += 1

        # Submit one or two reads and measure latency
        reads_this_batch = min(2, num_reads - len(read_latencies))
        for i in range(reads_this_batch):
            file_idx = (batch_idx + i) % num_read_files
            block_ids = list(
                range(file_idx * blocks_per_file, (file_idx + 1) * blocks_per_file)
            )
            read_gpu = make_gpu_specs(block_ids)
            read_storage = SharedStorageLoadStoreSpec(
                [read_put_storage.block_hashes[file_idx]]
            )

            read_job_id = 1000 + len(read_latencies)

            start = time.time()
            get.transfer_async(job_id=read_job_id, spec=(read_storage, read_gpu))
            ok = wait_for(
                get, job_id=read_job_id, timeout=30.0, _finished_cache=finished_cache
            )
            latency = time.time() - start

            assert ok, f"Read job {read_job_id} failed"
            read_latencies.append(latency)

        # Small delay between batches
        time.sleep(0.02)

    # Wait for all writes to complete
    for job_id in range(1, write_job_id):
        wait_for(put, job_id=job_id, timeout=60.0, _finished_cache=finished_cache)

    # Step 3: Calculate percentiles
    read_latencies.sort()
    n = len(read_latencies)

    def percentile(data, p):
        idx = int(n * p / 100)
        return data[min(idx, n - 1)]

    p50 = percentile(read_latencies, 50)
    p95 = percentile(read_latencies, 95)
    p99 = percentile(read_latencies, 99)
    p_min = read_latencies[0]
    p_max = read_latencies[-1]
    tail_ratio = p99 / p50 if p50 > 0 else float("inf")

    print(f"\n{'=' * 70}")
    print("Latency Percentiles Test Results")
    print(f"{'=' * 70}")
    print("Configuration:")
    print(f"  Threads: {threads_per_gpu}")
    print(f"  Total writes submitted: {write_job_id - 1}")
    print(f"  Total reads measured: {n}")
    print("\nRead Latency Distribution:")
    print(f"  Min:  {p_min:.3f}s")
    print(f"  p50:  {p50:.3f}s (median)")
    print(f"  p95:  {p95:.3f}s")
    print(f"  p99:  {p99:.3f}s (worst case)")
    print(f"  Max:  {p_max:.3f}s")
    print("\nQuality of Service:")
    print(f"  Tail ratio (p99/p50): {tail_ratio:.2f}x")
    print("  Expected with good priority: <3.0x")
    print(f"{'=' * 70}")

    # Assertions
    # p99 should be reasonable even under write pressure
    assert p99 < 1.0, (
        f"p99 read latency too high: {p99:.3f}s (expected <1.0s). "
        f"Reads are experiencing excessive queuing delays."
    )

    # Tail latency shouldn't be much worse than median
    assert tail_ratio < 5.0, (
        f"High tail latency ratio: {tail_ratio:.2f}x (p99/p50). "
        f"Expected <5.0x. This indicates inconsistent read performance - "
        f"some reads are waiting much longer than others."
    )

    # Cleanup
    cleanup_files(file_mapper, read_hashes)
    for i in range(write_file_idx):
        _, write_hashes = make_storage_specs(1, start_offset=write_offset + i)
        cleanup_files(file_mapper, write_hashes)

    del handler, put, get


def test_sustained_read_latency_under_write_saturation(default_vllm_config):
    """
    Test read latency when write queue is continuously saturated.

    This simulates a realistic production scenario:
    - Heavy write traffic (cache evictions during high load)
    - Occasional reads (cache hits for inference)

    Strategy:
        1. Saturate queue with continuous writes
        2. Submit reads at regular intervals
        3. Verify reads maintain low latency

    Success criteria:
        - All reads complete successfully
        - Read latency remains bounded
        - Writes don't starve reads
    """
    threads_per_gpu = 8
    num_write_files = 40
    num_read_operations = 15

    blocks_per_file = TEST_CONFIG["gpu_blocks_per_file"]
    num_read_files = 5
    num_blocks = (num_read_files + num_write_files) * blocks_per_file

    handler, context = create_test_handler(
        num_blocks=num_blocks,
        threads_per_gpu=threads_per_gpu,
        model_suffix="-saturation",
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

    # Submit all writes at once to saturate queue
    write_offset = num_read_files
    write_block_start = num_read_files * blocks_per_file

    print(f"\nSubmitting {num_write_files} write operations to saturate queue...")
    write_job_ids = []
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
        write_job_ids.append(job_id)

    # Now continuously submit reads and measure latency
    read_latencies = []
    max_latency = 0

    print(f"Submitting {num_read_operations} reads while writes are processing...")
    for i in range(num_read_operations):
        file_idx = i % num_read_files
        block_ids = list(
            range(file_idx * blocks_per_file, (file_idx + 1) * blocks_per_file)
        )
        read_gpu = make_gpu_specs(block_ids)
        read_storage = SharedStorageLoadStoreSpec(
            [read_put_storage.block_hashes[file_idx]]
        )

        read_job_id = 1000 + i

        start = time.time()
        get.transfer_async(job_id=read_job_id, spec=(read_storage, read_gpu))
        ok = wait_for(
            get, job_id=read_job_id, timeout=30.0, _finished_cache=finished_cache
        )
        latency = time.time() - start

        assert ok, f"Read {i} failed"
        read_latencies.append(latency)
        max_latency = max(max_latency, latency)

        time.sleep(0.05)

    avg_latency = sum(read_latencies) / len(read_latencies)

    print(f"\n{'=' * 70}")
    print("Sustained Read Latency Test Results")
    print(f"{'=' * 70}")
    print("Configuration:")
    print(f"  Threads: {threads_per_gpu}")
    print(f"  Write saturation: {num_write_files} files")
    print(f"  Read operations: {num_read_operations}")
    print("\nRead Latency Under Saturation:")
    print(f"  Average: {avg_latency:.3f}s")
    print(f"  Maximum: {max_latency:.3f}s")
    print(f"  All reads completed: {len(read_latencies)}/{num_read_operations}")
    print(f"{'=' * 70}")

    # Reads should maintain reasonable latency even under saturation
    assert max_latency < 1.5, (
        f"Max read latency {max_latency:.3f}s exceeded threshold (1.5s) "
        f"under write saturation. Reads are being starved by writes."
    )

    assert avg_latency < 0.5, (
        f"Average read latency {avg_latency:.3f}s exceeded threshold (0.5s). "
        f"Priority is not effectively reducing read wait time."
    )

    # Wait for remaining writes
    for job_id in write_job_ids:
        wait_for(put, job_id=job_id, timeout=60.0, _finished_cache=finished_cache)

    # Cleanup
    cleanup_files(file_mapper, read_hashes)
    for i in range(num_write_files):
        _, write_hashes = make_storage_specs(1, start_offset=write_offset + i)
        cleanup_files(file_mapper, write_hashes)

    del handler, put, get


if __name__ == "__main__":
    # Allow running individual tests
    import sys

    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name == "order":
            test_priority_completion_order()
        elif test_name == "percentiles":
            test_read_latency_percentiles()
        elif test_name == "saturation":
            test_sustained_read_latency_under_write_saturation()
        else:
            print(f"Unknown test: {test_name}")
            print("Available tests: order, percentiles, saturation")
    else:
        print("Running all priority queue tests...")
        test_priority_completion_order()
        test_read_latency_percentiles()
        test_sustained_read_latency_under_write_saturation()
        print("\nAll tests passed!")
