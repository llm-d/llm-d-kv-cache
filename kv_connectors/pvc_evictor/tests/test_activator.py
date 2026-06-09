"""Unit tests for activator process configuration handling."""

import multiprocessing
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from processes.activator import activator_process  # noqa: E402


def test_activator_uses_config_log_level():
    """Activator must honour config_dict["log_level"], not hardcode 'INFO'."""
    with (
        patch("processes.activator.setup_logging") as mock_setup_logging,
        patch("processes.activator.get_disk_usage_from_statvfs"),
    ):
        shutdown_event = multiprocessing.Event()
        shutdown_event.set()  # exit loop immediately

        config_dict = {"log_level": "DEBUG", "log_file_path": None}
        activator_process(
            process_num=9,
            mount_path="/tmp",
            cleanup_threshold=85.0,
            target_threshold=70.0,
            logger_interval=0.5,
            deletion_event=multiprocessing.Event(),
            result_queue=multiprocessing.Queue(),
            shutdown_event=shutdown_event,
            config_dict=config_dict,
        )

        mock_setup_logging.assert_called_once_with("DEBUG", 9, None)


def test_activator_uses_config_log_file_path():
    """Activator must pass log_file_path from config_dict to setup_logging."""
    with (
        patch("processes.activator.setup_logging") as mock_setup_logging,
        patch("processes.activator.get_disk_usage_from_statvfs"),
    ):
        shutdown_event = multiprocessing.Event()
        shutdown_event.set()

        config_dict = {"log_level": "WARNING", "log_file_path": "/tmp/evictor.log"}
        activator_process(
            process_num=9,
            mount_path="/tmp",
            cleanup_threshold=85.0,
            target_threshold=70.0,
            logger_interval=0.5,
            deletion_event=multiprocessing.Event(),
            result_queue=multiprocessing.Queue(),
            shutdown_event=shutdown_event,
            config_dict=config_dict,
        )

        mock_setup_logging.assert_called_once_with("WARNING", 9, "/tmp/evictor.log")
