"""Recognize stale temporary files left behind by the vLLM filesystem KV tier.

vLLM writes each block to ``<dest>.bin_<random>.tmp`` and ``os.replace``s it
onto ``<dest>.bin``; at startup it probes O_DIRECT support with
``.o_direct_probe_<random>.tmp`` in the directory above the ``_r<rank>`` dirs.
If the engine is killed between create and rename (or during the probe), the
temp file is never removed.

Only these exact shapes are matched, so unrelated files on a shared PVC are
never selected. The age check uses modification time and cannot prove that the
writer is gone: a write stalled for longer than the threshold will fail if its
temp file is removed.
"""

import os
import re
import stat
import time

# <hash_hex>.bin_<non-negative decimal>.tmp (see vllm/v1/kv_offload/tiering/fs/io.py)
BLOCK_TMP_FILE_RE = re.compile(r"^[0-9a-fA-F]+\.bin_\d+\.tmp$")
# .o_direct_probe_<non-negative decimal>.tmp
PROBE_TMP_FILE_RE = re.compile(r"^\.o_direct_probe_\d+\.tmp$")


def is_block_tmp_file(name: str) -> bool:
    """Return whether a basename is a vLLM block temp file."""
    return BLOCK_TMP_FILE_RE.match(name) is not None


def is_probe_tmp_file(name: str) -> bool:
    """Return whether a basename is a vLLM O_DIRECT probe temp file."""
    return PROBE_TMP_FILE_RE.match(name) is not None


def is_stale_tmp_file(path: str, min_age_seconds: float, now: float | None = None) -> bool:
    """Return whether ``path`` is a regular file last modified at least ``min_age_seconds`` ago.

    Returns False when the file cannot be stat'ed (e.g. it was already renamed
    or removed) or is not a regular file, so callers never act on it.
    """
    if min_age_seconds <= 0:
        return False
    try:
        st = os.stat(path, follow_symlinks=False)
    except OSError:
        return False
    if not stat.S_ISREG(st.st_mode):
        return False
    if now is None:
        now = time.time()
    return now - st.st_mtime >= min_age_seconds
