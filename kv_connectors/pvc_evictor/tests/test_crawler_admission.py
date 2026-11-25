"""Exercise backpressure with live producer/consumer threads and lazy traversal."""

import os
import queue
import threading
from pathlib import Path

import pytest
from processes import crawler
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


def test_full_prefill_does_not_advance_and_shutdown_interrupts_pause(monkeypatch):
    # A normal sleep would outlive join(1); Event.wait must wake on shutdown.
    monkeypatch.setattr(crawler, "QUEUE_FULL_SLEEP_SECONDS", 60)
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


def test_crawler_process_pauses_walk_reports_stats_and_resumes(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate.bin"
    candidate.write_bytes(b"cache")
    os.utime(candidate, (1, 1))
    files, results = queue.Queue(maxsize=2), queue.Queue()
    files.put("prefilled")
    pressure, stop, advanced = threading.Event(), threading.Event(), threading.Event()

    def walk(*args, **kwargs):
        advanced.set()
        yield candidate

    monkeypatch.setattr(crawler, "stream_cache_files_with_mapper", walk)
    monkeypatch.setattr(crawler, "setup_logging", lambda *args: None)
    monkeypatch.setattr(crawler, "AGGREGATED_LOGGING_INTERVAL_SECONDS", 0)
    thread = threading.Thread(
        target=crawler.crawler_process,
        args=(
            0,
            (0, 15),
            tmp_path,
            {
                "file_queue_min_size": 1,
                "file_queue_maxsize": 2,
                "file_access_time_threshold_minutes": 60,
            },
            pressure,
            files,
            results,
            stop,
        ),
    )
    thread.start()
    try:
        kind, _, stats = results.get(timeout=1)
        assert kind == "crawler_stats" and stats["queue_size"] == 1
        assert not advanced.is_set()
        pressure.set()
        assert advanced.wait(1)
        assert files.get(timeout=1) == "prefilled"
        assert files.get(timeout=1) == str(candidate)
    finally:
        stop.set()
        thread.join(1)
    assert not thread.is_alive()


@pytest.mark.parametrize("race", ["idle_filled", "active_full", "turned_off"])
def test_crawler_retains_candidate_when_enqueue_loses_race(tmp_path, monkeypatch, race):
    candidate = tmp_path / "candidate.bin"
    candidate.write_bytes(b"cache")
    os.utime(candidate, (1, 1))
    pressure, stop = threading.Event(), threading.Event()
    waiting, enqueued = threading.Event(), threading.Event()
    if race != "idle_filled":
        pressure.set()

    class RacingQueue:
        full = False
        attempts = 0
        items = []

        def qsize(self):
            if self.full:
                waiting.set()
            return 1 if self.full else 0

        def put(self, item, timeout):
            self.attempts += 1
            if self.full:
                waiting.set()
                assert timeout <= 0.1
                stop.wait(timeout)
                raise queue.Full
            self.items.append(item)
            enqueued.set()
            stop.set()

    files = RacingQueue()

    def walk(*args, **kwargs):
        # Admission observed capacity, but another producer fills it before put.
        files.full = True
        if race != "active_full":
            pressure.clear()
        yield candidate

    monkeypatch.setattr(crawler, "stream_cache_files_with_mapper", walk)
    monkeypatch.setattr(crawler, "setup_logging", lambda *args: None)
    thread = threading.Thread(
        target=crawler.crawler_process,
        args=(
            0,
            (0, 15),
            tmp_path,
            {
                "file_queue_min_size": 1,
                "file_queue_maxsize": 2,
                "file_access_time_threshold_minutes": 60,
            },
            pressure,
            files,
            queue.Queue(),
            stop,
        ),
    )
    thread.start()
    try:
        assert waiting.wait(1)
        assert not enqueued.is_set()
        files.full = False
        assert enqueued.wait(1)
    finally:
        stop.set()
        thread.join(1)
    assert files.items == [str(candidate)]
    assert not thread.is_alive()
