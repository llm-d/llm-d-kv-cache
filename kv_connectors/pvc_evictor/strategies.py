# Copyright 2026 The llm-d Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import time
import logging
import threading
import json
import multiprocessing
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional

try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False

try:
    from llmd_fs_backend.file_mapper import FileMapper
    FILEMAPPER_AVAILABLE = True
except ImportError:
    FILEMAPPER_AVAILABLE = False

logger = logging.getLogger("evictor.strategies")

# =====================================================================
# 1. Eviction Policy Strategy Interface & Implementations
# =====================================================================

class EvictionPolicy(ABC):
    @abstractmethod
    def record_file_discovery(self, file_path: str, stat_info: Optional[Tuple[float, float]] = None) -> None:
        """Record file discovered via disk scan. stat_info is (atime, mtime)."""
        pass

    @abstractmethod
    def record_file_creation(self, file_path: str, creation_time: float) -> None:
        """Record file creation event."""
        pass

    @abstractmethod
    def record_file_access(self, file_path: str, access_time: float) -> None:
        """Record file access event."""
        pass

    @abstractmethod
    def record_file_deletion(self, file_path: str) -> None:
        """Record manual/vLLM deletion event."""
        pass

    @abstractmethod
    def get_eviction_candidates(self, batch_size: int) -> List[str]:
        """Get the best candidates to delete next."""
        pass


class AgeTTLPolicy(EvictionPolicy):
    """Original FIFO/TTL deletion policy."""
    def __init__(self, ttl_minutes: float):
        self.ttl_seconds = ttl_minutes * 60.0
        self.files: Dict[str, float] = {}  # path -> last_active_time
        self.lock = threading.Lock()

    def record_file_discovery(self, file_path: str, stat_info: Optional[Tuple[float, float]] = None) -> None:
        with self.lock:
            if file_path not in self.files:
                active_time = time.time()
                if stat_info:
                    active_time = max(stat_info[0], stat_info[1])
                self.files[file_path] = active_time

    def record_file_creation(self, file_path: str, creation_time: float) -> None:
        with self.lock:
            self.files[file_path] = creation_time

    def record_file_access(self, file_path: str, access_time: float) -> None:
        with self.lock:
            self.files[file_path] = access_time

    def record_file_deletion(self, file_path: str) -> None:
        with self.lock:
            self.files.pop(file_path, None)

    def get_eviction_candidates(self, batch_size: int) -> List[str]:
        candidates = []
        now = time.time()
        with self.lock:
            for file_path, active_time in list(self.files.items()):
                if len(candidates) >= batch_size:
                    break
                if self.ttl_seconds > 0 and (now - active_time < self.ttl_seconds):
                    continue
                candidates.append(file_path)
            for p in candidates:
                self.files.pop(p, None)
        return candidates


class LRUPolicy(EvictionPolicy):
    """Least-Recently-Used (LRU) eviction policy with startup filesystem hydration."""
    def __init__(self, ttl_minutes: float):
        self.ttl_seconds = ttl_minutes * 60.0
        self.lru_cache: OrderedDict[str, float] = OrderedDict()  # path -> access_time
        self.lock = threading.Lock()
        self._dirty = False

    def record_file_discovery(self, file_path: str, stat_info: Optional[Tuple[float, float]] = None) -> None:
        """
        Hydrates pre-existing files discovered during the initial disk sweep.
        Inserts them into the LRU cache. To ensure pre-existing files are sorted correctly by age,
        they are registered with their filesystem access/modification time.
        """
        with self.lock:
            if file_path not in self.lru_cache:
                active_time = time.time()
                if stat_info:
                    active_time = max(stat_info[0], stat_info[1])
                self.lru_cache[file_path] = active_time
                # Mark as dirty so we perform a single aggregated sort before eviction
                self._dirty = True

    def record_file_creation(self, file_path: str, creation_time: float) -> None:
        """Record file creation event (e.g. BLOCK_CREATED). Moves/adds to MRU (end)."""
        with self.lock:
            self.lru_cache[file_path] = creation_time
            self.lru_cache.move_to_end(file_path, last=True)  # Move to the most-recently-used end (right)

    def record_file_access(self, file_path: str, access_time: float) -> None:
        """
        Record file access event (e.g. BLOCK_ACCESSED).
        Reorders the file, moving it to the most-recently-used end (right) of the queue.
        """
        with self.lock:
            self.lru_cache[file_path] = access_time
            self.lru_cache.move_to_end(file_path, last=True)  # Reorder: move to MRU (right)

    def record_file_deletion(self, file_path: str) -> None:
        with self.lock:
            self.lru_cache.pop(file_path, None)

    def get_eviction_candidates(self, batch_size: int) -> List[str]:
        candidates = []
        now = time.time()
        with self.lock:
            if self._dirty:
                self._sort_by_age()
                self._dirty = False
            for file_path, active_time in list(self.lru_cache.items()):
                if len(candidates) >= batch_size:
                    break
                if self.ttl_seconds > 0 and (now - active_time < self.ttl_seconds):
                    continue
                candidates.append(file_path)
            for p in candidates:
                self.lru_cache.pop(p, None)
        return candidates

    def _sort_by_age(self) -> None:
        """Internal helper to keep the OrderedDict sorted by timestamp values (oldest first)."""
        sorted_items = sorted(self.lru_cache.items(), key=lambda item: item[1])
        self.lru_cache = OrderedDict(sorted_items)

# =====================================================================
# 2. File Discovery Strategy Interface & Implementations
# =====================================================================

class FileDiscoveryStrategy(ABC):
    @abstractmethod
    def start(self, result_queue: multiprocessing.Queue) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def is_active(self) -> bool:
        pass


class EventDrivenDiscovery(FileDiscoveryStrategy):
    """ZMQ PUB/SUB event receiver (supports JSON & MsgPack)."""
    def __init__(self, config: Any, policy: EvictionPolicy, cache_path: Path):
        self.config = config
        self.policy = policy
        self.cache_path = cache_path
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.mappers: List[Any] = []
        self.last_mapper_scan = 0.0

    def check_connectivity(self) -> bool:
        if not ZMQ_AVAILABLE:
            return False
        try:
            import socket
            from urllib.parse import urlparse
            endpoint = self.config.zmq_endpoint
            if endpoint.startswith("ipc://"):
                ipc_path = endpoint[6:]
                return os.path.exists(ipc_path)
            
            if "://" in endpoint:
                parsed = urlparse(endpoint)
                host = parsed.hostname or "localhost"
                port = parsed.port or 5557
            else:
                host, port_str = endpoint.split(":")
                port = int(port_str)
            
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except Exception:
            return False

    def _refresh_mappers(self) -> None:
        if not FILEMAPPER_AVAILABLE:
            logger.warning("EventDiscovery: FileMapper class is not available. Hash mapping disabled.")
            return
        
        new_mappers = []
        try:
            if not self.cache_path.exists():
                return
            for model_dir in self.cache_path.iterdir():
                if not model_dir.is_dir():
                    continue
                for config_dir in model_dir.glob("block_size_*_blocks_per_file_*"):
                    params = self._parse_params(config_dir.name, "block_size_{gpu_block_size}_blocks_per_file_{gpu_blocks_per_file}")
                    if not params:
                        continue
                    for par_dir in config_dir.glob("tp_*_pp_size_*_pcp_size_*"):
                        par_params = self._parse_params(par_dir.name, "tp_{tp_size}_pp_size_{pp_size}_pcp_size_{pcp_size}")
                        if not par_params:
                            continue
                        for rank_dir in par_dir.glob("rank_*"):
                            rank_match = re.match(r"rank_(\d+)", rank_dir.name)
                            if not rank_match:
                                continue
                            rank = int(rank_match.group(1))
                            for dtype_dir in rank_dir.iterdir():
                                if not dtype_dir.is_dir():
                                    continue
                                
                                mapper = FileMapper(
                                    root_dir=str(self.cache_path),
                                    model_name=model_dir.name,
                                    gpu_block_size=params["gpu_block_size"],
                                    gpu_blocks_per_file=params["gpu_blocks_per_file"],
                                    tp_size=par_params["tp_size"],
                                    pp_size=par_params["pp_size"],
                                    pcp_size=par_params["pcp_size"],
                                    rank=rank,
                                    dtype=dtype_dir.name
                                )
                                new_mappers.append(mapper)
            self.mappers = new_mappers # Atomic Reference Swap (Thread-safe assignment)
            logger.info(f"EventDiscovery: Registered {len(self.mappers)} active FileMappers from disk.")
        except Exception as e:
            logger.error(f"EventDiscovery: Error discovering FileMappers: {e}")

    def _parse_params(self, dir_name: str, pattern: str) -> Dict[str, int]:
        regex_pattern = pattern
        param_names = re.findall(r"\{(\w+)\}", pattern)
        for param in param_names:
            regex_pattern = regex_pattern.replace(f"{{{param}}}", f"(?P<{param}>\\d+)")
        match = re.match(regex_pattern, dir_name)
        if not match:
            return {}
        return {param: int(match.group(param)) for param in param_names}

    def _map_hash_to_file(self, model_name: str, rank: Optional[int], block_hash: Any) -> Optional[str]:
        if isinstance(block_hash, bytes):
            hash_hex = block_hash.hex()
        elif isinstance(block_hash, int):
            hash_hex = format(block_hash, '016x')
        else:
            hash_hex = str(block_hash)

        mapped_file = self._do_map_hash(model_name, rank, hash_hex)
        if mapped_file:
            return mapped_file
            
        # Rate-limited dynamic mapper refresh if mapping fails (e.g. new rank just spun up)
        now = time.time()
        if now - self.last_mapper_scan > 10.0:
            logger.info(f"Mapping failed for hash {hash_hex[:10]}. Triggering rate-limited mapper sweep...")
            self._refresh_mappers()
            self.last_mapper_scan = now
            return self._do_map_hash(model_name, rank, hash_hex)
            
        return None

    def _do_map_hash(self, model_name: str, rank: Optional[int], hash_hex: str) -> Optional[str]:
        subfolder1, subfolder2 = hash_hex[:3], hash_hex[3:5]
        for mapper in self.mappers:
            if model_name not in mapper.base_path:
                continue
            if rank is not None and f"/rank_{rank}/" not in mapper.base_path:
                continue
            
            # Dynamically search parent subfolder1 to check any group suffix (e.g. _g0, _g1, _g2)
            parent_dir = Path(mapper.base_path) / subfolder1
            if parent_dir.exists():
                try:
                    for child in parent_dir.iterdir():
                        if child.is_dir() and child.name.startswith(f"{subfolder2}_g"):
                            file_path = child / f"{hash_hex}.bin"
                            if file_path.exists():
                                return str(file_path)
                except OSError:
                    pass
        return None

    def start(self, result_queue: multiprocessing.Queue) -> None:
        if self.running:
            return
        self.running = True
        self._refresh_mappers()
        self.last_mapper_scan = time.time()
        self.thread = threading.Thread(target=self._subscriber_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False

    def is_active(self) -> bool:
        return self.running

    def _subscriber_loop(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect(self.config.zmq_endpoint)
        socket.setsockopt_string(zmq.SUBSCRIBE, self.config.zmq_topic)
        socket.setsockopt(zmq.RCVTIMEO, 1000)

        logger.info(f"ZMQ Event Subscriber active on topic: {self.config.zmq_topic}")

        while self.running:
            try:
                now = time.time()
                # Scan/refresh file mappers on disk every 5 minutes to pickup new model workers dynamically
                if now - self.last_mapper_scan > 300.0:
                    self._refresh_mappers()
                    self.last_mapper_scan = now

                parts = socket.recv_multipart()
                if len(parts) < 3:
                    continue

                topic = parts[0].decode("utf-8", errors="ignore")
                payload = parts[2]
                
                topic_parts = topic.split("@")
                model_name = topic_parts[2] if len(topic_parts) >= 3 else ""

                if payload.startswith(b"{"):
                    # Parse JSON Simulator Event
                    data = json.loads(payload.decode("utf-8", errors="ignore"))
                    for ev in data.get("events", []):
                        file_path = ev.get("file_path")
                        ev_type = ev.get("type")
                        if file_path:
                            if ev_type == "BLOCK_CREATED":
                                self.policy.record_file_creation(file_path, now)
                            elif ev_type == "BLOCK_ACCESSED":
                                self.policy.record_file_access(file_path, now)
                else:
                    # Parse MsgPack vLLM Production Event
                    if MSGPACK_AVAILABLE:
                        event_dict = msgpack.unpackb(payload, raw=False)
                        rank = event_dict.get("data_parallel_rank")
                        for ev in event_dict.get("events", []):
                            if "block_hashes" in ev:
                                block_hashes = ev["block_hashes"]
                                if "parent_block_hash" in ev or "token_ids" in ev:
                                    # BlockStored (Created)
                                    for bh in block_hashes:
                                        path = self._map_hash_to_file(model_name, rank, bh)
                                        if path:
                                            self.policy.record_file_creation(path, now)
                                else:
                                    # BlockRemoved (Deleted)
                                    for bh in block_hashes:
                                        path = self._map_hash_to_file(model_name, rank, bh)
                                        if path:
                                            self.policy.record_file_deletion(path)
            except zmq.Again:
                continue
            except Exception as e:
                logger.error(f"ZMQ subscriber error: {e}")
                time.sleep(1.0)
        socket.close()
        context.term()


class DiscoveryCoordinator(FileDiscoveryStrategy):
    """Orchestrates scanning & event discovery with dynamic connection/fallback."""
    def __init__(self, config: Any, policy: EvictionPolicy, cache_path: Path, spawn_scanner_fn: Any):
        self.config = config
        self.policy = policy
        self.cache_path = cache_path
        self.spawn_scanner_fn = spawn_scanner_fn
        
        self.event_discovery = EventDrivenDiscovery(config, policy, cache_path)
        self.zmq_active = False
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None

    def start(self, result_queue: multiprocessing.Queue) -> None:
        self.running = True
        
        # 1. Start standard filesystem scanning pool (initially)
        self.spawn_scanner_fn()
        
        # 2. Start ZMQ status monitor thread
        if self.config.kv_events_enabled:
            self.monitor_thread = threading.Thread(target=self._zmq_monitor_loop, args=(result_queue,), daemon=True)
            self.monitor_thread.start()

    def stop(self) -> None:
        self.running = False
        self.event_discovery.stop()

    def is_active(self) -> bool:
        return self.zmq_active or self.event_discovery.is_active()

    def _zmq_monitor_loop(self, result_queue: multiprocessing.Queue) -> None:
        logger.info("ZMQ Connection Monitor thread active.")
        while self.running:
            is_connected = self.event_discovery.check_connectivity()
            if not self.zmq_active and is_connected:
                logger.warning("ZMQ Endpoint Reachable! Activating ZMQ Event Driven Discovery.")
                self.event_discovery.start(result_queue)
                self.zmq_active = True
                result_queue.put(("zmq_active", True))
            elif self.zmq_active and not is_connected:
                logger.warning("ZMQ Endpoint Lost! Falling back to standard filesystem crawler.")
                self.event_discovery.stop()
                self.zmq_active = False
                result_queue.put(("zmq_active", False))
            time.sleep(15.0)

# =====================================================================
# 3. Continuous Absolute Age Purge (Background Daemon)
# =====================================================================

class ContinuousCleanup:
    """Deletes files older than absolute threshold hours, regardless of disk usage."""
    def __init__(self, config: Any, cache_path: Path, policy: Optional[EvictionPolicy] = None):
        self.config = config
        self.cache_path = cache_path
        self.policy = policy
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.thread.start()
        logger.info(f"Continuous absolute purge daemon started (threshold: {self.config.continuous_cleanup_hours}h).")

    def stop(self) -> None:
        self.running = False

    def _cleanup_loop(self) -> None:
        interval_seconds = self.config.continuous_cleanup_interval_minutes * 60.0
        threshold_seconds = self.config.continuous_cleanup_hours * 3600.0
        
        while self.running:
            try:
                logger.info("Continuous Purge cycle started...")
                now = time.time()
                deleted = 0
                
                for root, _, files in os.walk(str(self.cache_path)):
                    if not self.running:
                        break
                    for f in files:
                        if f.endswith(".bin"):
                            f_path = os.path.join(root, f)
                            try:
                                stat = os.stat(f_path)
                                age = now - max(stat.st_mtime, stat.st_atime)
                                if age > threshold_seconds:
                                    os.unlink(f_path)
                                    deleted += 1
                                    if self.policy:
                                        self.policy.record_file_deletion(f_path)
                            except OSError:
                                pass
                if deleted > 0:
                    logger.warning(f"Continuous Purge: Cleaned up {deleted} old cache files (> {self.config.continuous_cleanup_hours} hours).")
            except Exception as e:
                logger.error(f"Error in continuous purge run: {e}")
            
            time.sleep(interval_seconds)
