"""Unit tests for the PVCEvictor monitoring loop."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import evictor as evictor_module  # noqa: E402
from config import Config  # noqa: E402


class FakeProcess:
    """Stand-in for multiprocessing.Process that never forks."""

    instances: list["FakeProcess"] = []

    def __init__(self, target=None, args=(), name=None):
        self.target = target
        self.args = args
        self.name = name
        self.exitcode = None
        # The first activator dies right away, as if it was OOM-killed.
        first_activator = name.startswith("Activator") and not any(
            p.name.startswith("Activator") for p in FakeProcess.instances
        )
        self.alive = not first_activator
        FakeProcess.instances.append(self)

    def start(self):
        pass

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        pass

    def terminate(self):
        self.alive = False


class BusyResultQueue:
    """Result queue that always has a message, like during active deletion."""

    def __init__(self, evictor, max_messages):
        self.evictor = evictor
        self.remaining = max_messages

    def get(self, timeout=None):
        self.remaining -= 1
        if self.remaining <= 0:
            self.evictor.running = False
        return ("progress", 100, 1024)


def test_dead_activator_restarted_while_results_keep_arriving(tmp_path, monkeypatch):
    """A steady stream of results must not starve the child process health check."""
    monkeypatch.setenv("PVC_MOUNT_PATH", str(tmp_path))
    monkeypatch.setenv("NUM_CRAWLER_PROCESSES", "1")
    monkeypatch.setenv("ENABLE_DIR_CLEANUP", "false")
    monkeypatch.setattr(evictor_module.signal, "signal", lambda *_: None)
    evictor = evictor_module.PVCEvictor(Config.from_env())

    FakeProcess.instances = []
    monkeypatch.setattr(evictor_module.multiprocessing, "Process", FakeProcess)
    monkeypatch.setattr(evictor_module.time, "sleep", lambda _: None)
    monkeypatch.setattr(evictor_module, "PROCESS_HEALTH_CHECK_INTERVAL_SECONDS", 0.0, raising=False)
    evictor.result_queue = BusyResultQueue(evictor, max_messages=5)

    evictor.run()

    activators = [p for p in FakeProcess.instances if p.name.startswith("Activator")]
    assert len(activators) == 2, "dead activator was not restarted"
