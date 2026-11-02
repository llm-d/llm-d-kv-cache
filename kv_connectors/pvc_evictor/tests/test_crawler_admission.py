"""Exercise backpressure with live producer/consumer threads and lazy traversal."""

import queue
import threading
from pathlib import Path

from processes.crawler import admitted_files


def start_producer(capacity=2, prefill=2, minimum=2, maximum=4):
    files = queue.Queue(maxsize=capacity)
    for i in range(prefill):
        files.put(f"queued-{i}")
    pressure = threading.Event()
    stop = threading.Event()
    advanced = threading.Event()
    traversed = []

    def walk():
        for i in range(10):
            traversed.append(i)
            advanced.set()
            yield Path(f"candidate-{i}")

    def produce():
        for path in admitted_files(walk(), files.qsize, pressure, stop, minimum, maximum):
            files.put(path, timeout=1)

    thread = threading.Thread(target=produce)
    thread.start()
    return files, pressure, stop, advanced, traversed, thread


def test_full_prefill_does_not_advance_and_shutdown_interrupts_pause():
    files, pressure, stop, advanced, traversed, thread = start_producer()
    try:
        assert not advanced.wait(0.25)
        assert traversed == []
    finally:
        stop.set()
        thread.join(1)
    assert not thread.is_alive()
    assert files.qsize() == 2


def test_draining_prefill_resumes_same_walk_and_blocks_again():
    files, pressure, stop, advanced, traversed, thread = start_producer()
    try:
        assert not advanced.wait(0.15)
        files.get(timeout=1)
        assert advanced.wait(1)
        # Producer fills the newly available slot, then pauses before next().
        thread.join(0.2)
        assert traversed == [0]
        advanced.clear()
        files.get(timeout=1)
        # One more slot lets traversal progress without restarting at file zero.
        assert advanced.wait(1)
        assert traversed == [0, 1]
    finally:
        stop.set()
        thread.join(1)
    assert not thread.is_alive()


def test_pressure_increases_admission_then_off_restores_prefill_target():
    files, pressure, stop, advanced, traversed, thread = start_producer(capacity=4)
    try:
        assert not advanced.wait(0.15)
        pressure.set()
        assert advanced.wait(1)
        thread.join(0.2)
        assert traversed == [0, 1]
        pressure.clear()
        # Drain to MINQ: no further traversal until below that target.
        files.get(timeout=1)
        files.get(timeout=1)
        advanced.clear()
        assert not advanced.wait(0.2)
        files.get(timeout=1)
        assert advanced.wait(1)
        assert traversed == [0, 1, 2]
    finally:
        stop.set()
        thread.join(1)
    assert not thread.is_alive()
