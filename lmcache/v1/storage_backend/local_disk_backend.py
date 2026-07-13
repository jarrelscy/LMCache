# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence
import asyncio
import json
import os
import threading
import time

# Third Party
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    TORCH_DTYPE_TO_STR_DTYPE,
    CacheEngineKey,
    DiskCacheMetadata,
    _lmcache_nvtx_annotate,
)
from lmcache.v1.cache_controller.message import OpType
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.batched_message_sender import BatchedMessageSender
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.storage_backend.job_executor.pq_executor import (
    AsyncPQThreadPoolExecutor,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.path_sharder import PathSharder

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)

_DEFAULT_THREAD_COUNT = 4

# On-disk sidecar metadata format for LocalDiskBackend entries.
#
# LocalDiskBackend historically wrote only the raw KV bytes to
# `<key>.pt`, keeping shape/dtype/fmt/size solely in the in-memory
# `self.dict` index. That index was never rebuilt from disk on startup,
# so every process restart silently dropped 100% of the persisted cache
# (a `contains()` miss on every key, indistinguishable from an empty
# cache) even though the bytes were still sitting on disk. This sidecar
# (`<key>.pt.meta`) plus the rebuild/lazy-load logic below fixes that.
_LOCAL_DISK_METADATA_FILE_SUFFIX = ".meta"
_LOCAL_DISK_METADATA_VERSION = 1


def _pack_local_disk_metadata(memory_obj: MemoryObj) -> bytes:
    """Serialize the (shape, dtype, fmt, size) needed to reconstruct a
    ``DiskCacheMetadata`` entry, so a later process can rebuild its index
    from the files already on disk."""
    tensor = memory_obj.tensor
    assert tensor is not None
    meta = {
        "lmcache_version": _LOCAL_DISK_METADATA_VERSION,
        "shape": list(tensor.size()),
        "dtype": TORCH_DTYPE_TO_STR_DTYPE[tensor.dtype],
        "fmt": memory_obj.metadata.fmt.value,
        "size": memory_obj.get_physical_size(),
    }
    return json.dumps(meta).encode("utf-8")


def _unpack_local_disk_metadata(buf: bytes):
    meta = json.loads(buf.decode("utf-8"))
    if meta.get("lmcache_version") != _LOCAL_DISK_METADATA_VERSION:
        raise ValueError(f"Unsupported local-disk metadata payload: {meta}")
    shape = torch.Size(meta["shape"])
    dtype = STR_DTYPE_TO_TORCH_DTYPE[meta["dtype"]]
    fmt = MemoryFormat(meta["fmt"])
    size = int(meta["size"])
    return shape, dtype, fmt, size


# TODO(Jiayi): handle cases where cache is repetitvely prefetched.
class LocalDiskWorker:
    def __init__(
        self, loop: asyncio.AbstractEventLoop, max_workers: int = _DEFAULT_THREAD_COUNT
    ) -> None:
        self.put_lock = threading.Lock()
        self.put_tasks: List[CacheEngineKey] = []

        self.prefetch_lock = threading.Lock()
        self.prefetch_tasks: dict[CacheEngineKey, Future] = {}

        self.executor = AsyncPQThreadPoolExecutor(loop, max_workers=max_workers)
        self.loop = loop
        self._closed = False

    async def submit_task(
        self,
        task_type: str,
        task: Callable,
        *args,
        **kwargs,
    ) -> Any:
        if task_type == "prefetch":
            priority = 0
            # self.insert_prefetch_task(kwargs["key"], None)
        elif task_type == "delete":
            priority = 1
        elif task_type == "put":
            priority = 2
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        return await self.executor.submit_job(
            task,
            *args,
            priority=priority,
            **kwargs,
        )

    def remove_put_task(self, key: CacheEngineKey):
        with self.put_lock:
            if key in self.put_tasks:
                self.put_tasks.remove(key)
            else:
                logger.warning(f"Key {key} not found in put tasks.")

    def insert_put_task(self, key: CacheEngineKey):
        with self.put_lock:
            self.put_tasks.append(key)

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.put_lock:
            return key in self.put_tasks

    def close(self):
        # Gracefully shut down the executor
        if self._closed:
            return
        self._closed = True
        self.executor.shutdown(wait=True)


class LocalDiskBackend(StorageBackendInterface):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        dst_device: str = torch_device_type,
        lmcache_worker: Optional["LMCacheWorker"] = None,
        metadata: Optional[LMCacheMetadata] = None,
    ):
        if torch_dev.is_available():
            super().__init__(dst_device)
        else:
            super().__init__("cpu")

        self.cache_policy = get_cache_policy(config.cache_policy)
        self.dict = self.cache_policy.init_mutable_mapping()

        self.dst_device = dst_device

        self.local_cpu_backend = local_cpu_backend

        self.disk_lock = threading.Lock()

        assert config.local_disk is not None

        sharder = PathSharder(
            raw_csv=config.local_disk,
            strategy=config.local_disk_path_sharding,
            dst_device=dst_device,
            create_dirs=True,
        )
        self.path: str = sharder.selected

        logger.info(
            "Local disk cache path: %s (device %s, %d path(s) configured)",
            self.path,
            dst_device,
            len(sharder.all_paths),
        )

        self.loop = loop

        self.use_local_cpu = config.local_cpu

        # Block size (for file system I/O)
        stat = os.statvfs(self.path)
        self.os_disk_bs = stat.f_bsize
        self.use_odirect = False

        if config.extra_config is not None:
            self.use_odirect = config.extra_config.get("use_odirect", False)
        logger.info("Using O_DIRECT for disk I/O: %s", self.use_odirect)

        thread_count = config.get_extra_config_value(
            "disk_io_threads", _DEFAULT_THREAD_COUNT
        )
        self.disk_worker = LocalDiskWorker(loop, max_workers=thread_count)

        # Plain ThreadPoolExecutor for batched_get_blocking (concurrent
        # synchronous reads).  The existing AsyncPQThreadPoolExecutor in
        # disk_worker is async/priority-queue based and only serves writes
        # and prefetches; a simple pool is a better fit for the blocking
        # read path where we want ThreadPoolExecutor.map() semantics.
        self._read_thread_pool = ThreadPoolExecutor(
            max_workers=thread_count,
            thread_name_prefix="disk-read",
        )

        # TODO(Jiayi): We need a disk space allocator to avoid fragmentation
        # and hide the following details away from the backend.
        self.max_cache_size = int(config.max_local_disk_size * 1024**3)
        self.current_cache_size = 0.0

        # to help maintain suffix -> prefix order in the dict
        # assumption: only one request is looked up at a time
        # (only one worker per cache engine)
        self.keys_in_request: List[CacheEngineKey] = []

        self.lmcache_worker = lmcache_worker
        self.instance_id = config.lmcache_instance_id
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        self.usage = 0

        # Batched message sender for controller communication
        self.batched_msg_sender: Optional[BatchedMessageSender] = None

        # Initialize batched message sender
        if lmcache_worker and metadata is not None:
            self.batched_msg_sender = BatchedMessageSender(
                metadata=metadata,
                config=config,
                location=str(self),
                lmcache_worker=lmcache_worker,
            )
        else:
            logger.warning("Controller message sender is not initialized")

        # Rebuild the index from whatever is already on disk (e.g. from a
        # previous process instance / container restart). Without this,
        # `self.dict` starts empty every boot and every `contains()` call
        # reports a miss regardless of what is actually persisted on disk,
        # so cross-restart persistence is 0% even though the bytes are
        # there. See `_rebuild_index` / `_try_lazy_load_metadata` below.
        self.metadata = metadata
        self._index_rebuilt = threading.Event()
        self._rebuild_index_future = asyncio.run_coroutine_threadsafe(
            self._rebuild_index(), self.loop
        )

    def __str__(self) -> str:
        return "LocalDiskBackend"

    def _key_to_path(
        self,
        key: CacheEngineKey,
    ) -> str:
        return os.path.join(self.path, key.to_string().replace("/", "-") + ".pt")

    async def _rebuild_index(self) -> None:
        try:
            await asyncio.to_thread(self._rebuild_index_sync)
        except Exception:
            logger.exception(
                "LocalDiskBackend: startup index rebuild from %s failed; "
                "cache will behave as if empty (writes still work).",
                self.path,
            )
        finally:
            self._index_rebuilt.set()

    def _rebuild_index_sync(self) -> None:
        """Scan `self.path` for `<key>.pt` + `<key>.pt.meta` pairs left by a
        previous process instance and repopulate `self.dict` so that
        persisted entries survive a restart.

        Best-effort: any file that can't be parsed/decoded is skipped (and
        logged) rather than aborting the whole scan, and orphaned sidecar
        files (metadata written but the process crashed before/instead of
        finishing the data write, or vice versa) are cleaned up.
        """
        if self.metadata is None:
            logger.warning(
                "LocalDiskBackend: no LMCacheMetadata available, skipping "
                "startup index rebuild from %s (cross-restart persistence "
                "will not work this boot).",
                self.path,
            )
            return

        model_prefix = self.metadata.model_name.replace("/", "-")
        suffix = _LOCAL_DISK_METADATA_FILE_SUFFIX
        n_loaded = 0
        n_skipped = 0

        start = time.perf_counter()
        try:
            scan_iter = os.scandir(self.path)
        except OSError as e:
            logger.warning(
                "LocalDiskBackend: cannot scan %s for existing cache "
                "entries: %s",
                self.path,
                e,
            )
            return

        # Sidecar filenames are `<mangled_key_str>.pt.meta` (`_key_to_path`
        # appends ".pt" to the mangled key string, and the sidecar adds
        # `suffix` on top of that) -- strip both to recover the key string.
        data_suffix = ".pt" + suffix

        with scan_iter:
            for entry in scan_iter:
                if not entry.name.endswith(data_suffix) or not entry.is_file():
                    continue
                key_str = entry.name[: -len(data_suffix)]
                if not key_str.startswith(model_prefix + "@"):
                    # Either a stray file or an entry from a different
                    # model_name sharing this cache directory; not ours.
                    n_skipped += 1
                    continue
                unmangled_key_str = self.metadata.model_name + key_str[len(model_prefix):]
                data_path = entry.path[: -len(suffix)]
                if not os.path.exists(data_path):
                    # Orphan metadata (crash between the two writes below).
                    try:
                        os.remove(entry.path)
                    except OSError:
                        pass
                    n_skipped += 1
                    continue
                try:
                    key = CacheEngineKey.from_string(unmangled_key_str)
                    with open(entry.path, "rb") as f:
                        shape, dtype, fmt, size = _unpack_local_disk_metadata(
                            f.read()
                        )
                except Exception as e:
                    logger.warning(
                        "LocalDiskBackend: skipping unreadable cache entry "
                        "%s: %s",
                        entry.path,
                        e,
                    )
                    n_skipped += 1
                    continue
                with self.disk_lock:
                    if key not in self.dict:
                        self.dict[key] = DiskCacheMetadata(
                            data_path, size, shape, dtype, None, fmt, 0
                        )
                        self.current_cache_size += size
                        self.usage += size
                        n_loaded += 1

        self.stats_monitor.update_local_storage_usage(self.usage)
        logger.info(
            "LocalDiskBackend: rebuilt index from %s in %.2fs: "
            "%d entries recovered, %d skipped.",
            self.path,
            time.perf_counter() - start,
            n_loaded,
            n_skipped,
        )

    def _try_lazy_load_metadata(self, key: CacheEngineKey) -> bool:
        """Per-key fallback used on a `contains()`/`pin()` miss: the bulk
        `_rebuild_index` scan may not have reached this key yet (still
        running) or may have started after this key's files were written
        by a concurrent process. Mirrors GDS backend's lazy per-key probe.
        """
        path = self._key_to_path(key)
        meta_path = path + _LOCAL_DISK_METADATA_FILE_SUFFIX
        if not os.path.exists(meta_path) or not os.path.exists(path):
            return False
        try:
            with open(meta_path, "rb") as f:
                shape, dtype, fmt, size = _unpack_local_disk_metadata(f.read())
        except Exception as e:
            logger.warning(
                "LocalDiskBackend: lazy metadata read failed for %s at %s: %s",
                key,
                meta_path,
                e,
            )
            return False
        with self.disk_lock:
            if key not in self.dict:
                self.dict[key] = DiskCacheMetadata(path, size, shape, dtype, None, fmt, 0)
                self.current_cache_size += size
                self.usage += size
        return True

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self.disk_lock:
            found = key in self.dict
        if not found:
            found = self._try_lazy_load_metadata(key)
        if not found:
            return False
        with self.disk_lock:
            if pin:
                self.dict[key].pin()
                # vllm lookup sets pin to True
                self.keys_in_request.append(key)
            return True

    def touch_cache(self):
        # flip the order of the keys in the request
        with self.disk_lock:
            for key in reversed(self.keys_in_request):
                self.cache_policy.update_on_hit(key, self.dict)
            self.keys_in_request = []

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        return self.disk_worker.exists_in_put_tasks(key)

    def pin(
        self,
        key: CacheEngineKey,
    ) -> bool:
        with self.disk_lock:
            if key in self.dict:
                self.dict[key].pin()
                return True
            else:
                return False

    def unpin(
        self,
        key: CacheEngineKey,
    ) -> bool:
        with self.disk_lock:
            if key in self.dict:
                self.dict[key].unpin()
                return True
            else:
                return False

    def remove(
        self,
        key: CacheEngineKey,
        force: bool = True,
    ) -> bool:
        lock_context = self.disk_lock if force else nullcontext()
        with lock_context:
            if not (meta := self.dict.pop(key, None)):
                return False

            path = meta.path
            size = meta.size
            self.usage -= size
            self.stats_monitor.update_local_storage_usage(self.usage)

            # NOTE: The following code will cause deadlock
            # res = asyncio.run_coroutine_threadsafe(
            #     self.disk_worker.submit_task("delete", os.remove, path),
            #     self.loop,
            # )
            # res.result()

            os.remove(path)
            try:
                os.remove(path + _LOCAL_DISK_METADATA_FILE_SUFFIX)
            except OSError:
                pass

            if force:
                self.cache_policy.update_on_force_evict(key)

        # Push kv evict msg with batching
        if self.batched_msg_sender is not None:
            self.batched_msg_sender.add_kv_op(
                op_type=OpType.EVICT,
                key=key.chunk_hash,
            )

        return True

    def insert_key(
        self,
        key: CacheEngineKey,
        size: int,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat,
        cached_positions: Optional[torch.Tensor] = None,
    ) -> None:
        path = self._key_to_path(key)

        has_stored = False
        with self.disk_lock:
            if key in self.dict:
                # Update cache recency
                self.cache_policy.update_on_hit(key, self.dict)
                has_stored = True
            else:
                self.dict[key] = DiskCacheMetadata(
                    path, size, shape, dtype, cached_positions, fmt, 0
                )

        # Push kv admit msg with batching
        if self.batched_msg_sender is not None and not has_stored:
            self.batched_msg_sender.add_kv_op(
                op_type=OpType.ADMIT,
                key=key.chunk_hash,
            )

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ):
        """
        Submit a single put task to store KV cache to disk asynchronously.

        :param key: The cache key for this KV chunk.
        :param memory_obj: The memory object containing the KV data.
        :param on_complete_callback: Optional callback invoked once per key
            after the disk write completes. Callback exceptions are caught
            and logged.
        """
        assert memory_obj.tensor is not None

        # skip repeated save
        if self.exists_in_put_tasks(key):
            logger.debug(f"Put task for {key} is already in progress.")
            return None

        self.disk_worker.insert_put_task(key)

        # TODO(Jiayi): Fragmentation is not considered here.
        required_size = memory_obj.get_physical_size()
        all_evict_keys = []
        evict_success = True
        with self.disk_lock:
            while self.current_cache_size + required_size > self.max_cache_size:
                evict_keys = self.cache_policy.get_evict_candidates(
                    self.dict, num_candidates=1
                )
                if not evict_keys:
                    logger.warning(
                        "No eviction candidates found. Disk space under pressure."
                    )
                    evict_success = False
                    break

                for evict_key in evict_keys:
                    self.current_cache_size -= self.dict[evict_key].size

                self.batched_remove(evict_keys, force=False)

                all_evict_keys.extend(evict_keys)
            if evict_success:
                self.current_cache_size += required_size
                self.cache_policy.update_on_put(key)

        if not evict_success:
            return None

        memory_obj.ref_count_up()

        asyncio.run_coroutine_threadsafe(
            self.disk_worker.submit_task(
                "put",
                self.async_save_bytes_to_disk,
                key=key,
                memory_obj=memory_obj,
                on_complete_callback=on_complete_callback,
            ),
            self.loop,
        )

    # TODO(Jiayi): enable real batching
    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """
        Submit batched put tasks to store KV caches to disk asynchronously.

        :param keys: The cache keys for the KV chunks.
        :param memory_objs: The memory objects containing the KV data.
        :param transfer_spec: Optional transfer specification (unused).
        :param on_complete_callback: Optional callback invoked once per key
            after that key's disk write completes (not once per batch).
            Callback exceptions are caught and logged.
        """
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            self.submit_put_task(
                key, memory_obj, on_complete_callback=on_complete_callback
            )

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """
        Load a cached KV chunk from disk synchronously.

        The cache policy is updated only after a successful load so that a
        failed load (``load_bytes_from_disk`` returning ``None``) does not
        record a phantom cache hit and skew future eviction decisions.

        :param key: The cache key identifying the KV chunk.
        :returns: A ``MemoryObj`` containing the loaded KV data, or ``None``
            if the key is not present or the load fails.
        """
        with self.disk_lock:
            if key not in self.dict:
                return None

            disk_meta = self.dict[key]
            path = disk_meta.path
            dtype = disk_meta.dtype
            shape = disk_meta.shape
            fmt = disk_meta.fmt
            assert dtype is not None
            assert shape is not None

        # Load is performed outside the lock: it can block for a non-trivial
        # amount of time (CPU staging pool allocation + memcpy from disk) and
        # must not hold disk_lock while waiting, or concurrent insert/evict
        # operations would deadlock.
        memory_obj = self.load_bytes_from_disk(
            key, path, dtype=dtype, shape=shape, fmt=fmt
        )

        if memory_obj is not None:
            # Re-acquire the lock to update the eviction policy.  The key
            # membership check guards against the entry being evicted between
            # the two lock regions — in that case the policy state is already
            # consistent and no update is needed.
            with self.disk_lock:
                if key in self.dict:
                    self.cache_policy.update_on_hit(key, self.dict)

        return memory_obj

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        """Load multiple KV chunks from disk with concurrent I/O.

        Metadata lookup and memory allocation are performed sequentially
        under the disk lock, then all file reads are dispatched to a
        ``ThreadPoolExecutor`` so they run in parallel.  The GIL is
        released during the underlying ``readinto`` syscall, so threads
        achieve true I/O parallelism.

        :param keys: Cache keys identifying the KV chunks to load.
        :returns: A list of ``MemoryObj`` (or ``None`` for missing keys),
            in the same order as *keys*.
        """
        if len(keys) <= 1:
            return [self.get_blocking(k) for k in keys]

        # --- 1. Batch metadata lookup (single lock acquisition) -----------
        with self.disk_lock:
            metas = [self.dict.get(key) for key in keys]

        # --- 2. Pre-allocate staging buffers (sequential) -----------------
        memory_objs = [
            self.local_cpu_backend.allocate(m.shape, m.dtype, m.fmt)
            if m is not None
            else None
            for m in metas
        ]

        # --- 3. Concurrent file reads via thread pool ---------------------
        paths = [m.path if m is not None else None for m in metas]
        results: List[Optional[MemoryObj]] = list(
            self._read_thread_pool.map(
                self._load_chunk_into_memory, keys, paths, memory_objs
            )
        )

        # --- 4. Update cache policy for successful loads ------------------
        with self.disk_lock:
            for key, mem_obj in zip(keys, results, strict=True):
                if mem_obj is not None and key in self.dict:
                    self.cache_policy.update_on_hit(key, self.dict)

        return results

    def _load_chunk_into_memory(
        self,
        key: CacheEngineKey,
        path: Optional[str],
        memory_obj: Optional[MemoryObj],
    ) -> Optional[MemoryObj]:
        """Read a single chunk from disk into a pre-allocated ``MemoryObj``.

        Designed to be called from a thread pool — each invocation is
        independent and performs a single blocking ``readinto`` syscall.

        :param key: Cache key (used for metadata recovery and error logging).
        :param path: File path to read from, or ``None`` if the key was not
            found during the metadata lookup phase.
        :param memory_obj: Pre-allocated staging buffer, or ``None`` if
            allocation failed.
        :returns: The populated ``MemoryObj``, or ``None`` on any failure.
        """
        if path is None or memory_obj is None:
            return None

        try:
            buffer = memory_obj.byte_array
            self.read_file(key, buffer, path)

            # Recover metadata (mirrors load_bytes_from_disk).
            with self.disk_lock:
                disk_meta = self.dict.get(key)
                if disk_meta is None:
                    memory_obj.ref_count_down()
                    return None
                memory_obj.metadata.cached_positions = disk_meta.cached_positions

            return memory_obj
        except Exception as e:
            logger.error("Failed to load chunk from disk for key %s: %s", key, e)
            memory_obj.ref_count_down()
            return None

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        mem_objs: list[MemoryObj] = []
        paths: list[str] = []

        logger.debug(f"lookup_id: {lookup_id}; Prefetching {len(keys)} keys from disk.")
        for key in keys:
            self.disk_lock.acquire()
            assert key in self.dict, f"Key {key} not found in disk cache after pinning"

            path = self.dict[key].path
            dtype = self.dict[key].dtype
            shape = self.dict[key].shape
            fmt = self.dict[key].fmt

            assert dtype is not None
            assert shape is not None

            # busy_loop=False prevents spinning on the event loop thread;
            # if staging memory is exhausted the caller will get a logged
            # error rather than a silent deadlock.
            memory_obj = self.local_cpu_backend.allocate(
                shape,
                dtype,
                fmt,
                busy_loop=False,
            )

            if memory_obj is None:
                logger.error(
                    "Memory allocation failed during async disk load for key %s. "
                    "CPU staging pool may be exhausted (unpin() not called after "
                    "a previous retrieve). Returning partial results.",
                    key,
                )
                return mem_objs

            self.dict[key].pin()

            # NOTE(Jiayi): Currently, we consider prefetch as cache hit.
            # Update cache recency
            self.cache_policy.update_on_hit(key, self.dict)

            self.disk_lock.release()
            logger.debug(f"Prefetching {key} from disk.")
            memory_obj.pin()
            mem_objs.append(memory_obj)
            paths.append(path)

        return await self.disk_worker.submit_task(
            "prefetch",
            self.batched_async_load_bytes_from_disk,
            paths=paths,
            keys=keys,
            memory_objs=mem_objs,
        )

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        num_hit_counts = 0
        with self.disk_lock:
            for key in keys:
                if key not in self.dict:
                    return num_hit_counts
                if pin:
                    self.dict[key].pin()
                    self.keys_in_request.append(key)
                num_hit_counts += 1
        return num_hit_counts

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def async_save_bytes_to_disk(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """
        Convert KV to bytes and async store bytes to disk.

        :param on_complete_callback: Optional callback invoked after the disk
            write completes for this key. Callback exceptions are caught and
            logged.
        """
        kv_chunk = memory_obj.tensor
        assert kv_chunk is not None
        buffer = memory_obj.byte_array
        path = self._key_to_path(key)

        size = len(buffer)
        self.usage += size
        self.stats_monitor.update_local_storage_usage(self.usage)

        # TODO(Jiayi): need to add ref count in disk memory object
        self.write_file(buffer, path)

        # Persist a small sidecar with shape/dtype/fmt/size so a future
        # process instance can rebuild `self.dict` from disk (see
        # `_rebuild_index_sync`); without this, only the raw bytes survive
        # a restart and the in-memory index (the only place this metadata
        # otherwise lived) is gone, making every entry unreachable.
        try:
            meta_bytes = _pack_local_disk_metadata(memory_obj)
            self.write_file(meta_bytes, path + _LOCAL_DISK_METADATA_FILE_SUFFIX)
        except Exception as e:
            logger.warning(
                f"LocalDiskBackend: failed writing metadata sidecar for "
                f"{key} at {path}{_LOCAL_DISK_METADATA_FILE_SUFFIX}: {e}"
            )

        # ref count down here because there's a ref_count_up in
        # `submit_put_task` above.
        # Ref count down better be before `insert_key` for testing
        # purposes (e.g., testing mem_leak).
        # TODO(Jiayi): This could be problematic if the
        # freed memory object is immediately reused.
        size = memory_obj.get_physical_size()
        shape = memory_obj.metadata.shape
        dtype = memory_obj.metadata.dtype
        fmt = memory_obj.metadata.fmt
        cached_positions = memory_obj.metadata.cached_positions
        memory_obj.ref_count_down()

        self.insert_key(key, size, shape, dtype, fmt, cached_positions=cached_positions)

        self.disk_worker.remove_put_task(key)

        # Call the completion callback if provided
        if on_complete_callback is not None:
            try:
                on_complete_callback(key)
            except Exception as e:
                logger.warning(f"on_complete_callback failed for key {key}: {e}")

    @_lmcache_nvtx_annotate
    def batched_async_load_bytes_from_disk(
        self,
        paths: list[str],
        keys: list[CacheEngineKey],
        memory_objs: list[MemoryObj],
        write_back: bool = False,
    ) -> list[MemoryObj]:
        """
        Async load bytearray from disk.
        """

        logger.debug("Executing `async_load_bytes` from disk.")
        # TODO (Jiayi): handle the case where loading fails.
        for path, key, mem_obj in zip(paths, keys, memory_objs, strict=False):
            buffer = mem_obj.byte_array
            self.read_file(key, buffer, path)

            # TODO(Jiayi): Please recover the metadata in a more
            # elegant way in the future.
            cached_positions = self.dict[key].cached_positions
            mem_obj.metadata.cached_positions = cached_positions

            with self.disk_lock:
                self.dict[key].unpin()

        return memory_objs

    def load_bytes_from_disk(
        self,
        key: CacheEngineKey,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
        fmt: MemoryFormat,
    ) -> Optional[MemoryObj]:
        """
        Load bytearray from disk.
        """

        memory_obj = self.local_cpu_backend.allocate(shape, dtype, fmt)
        assert memory_obj is not None, "Memory allocation failed during disk load."

        buffer = memory_obj.byte_array
        self.read_file(key, buffer, path)

        # TODO(Jiayi): Please recover the metadata in a more
        # elegant way in the future.
        cached_positions = self.dict[key].cached_positions
        memory_obj.metadata.cached_positions = cached_positions

        return memory_obj

    def write_file(self, buffer, path):
        start_time = time.time()
        size = len(buffer)
        if size % self.os_disk_bs != 0 or not self.use_odirect:
            with open(path, "wb") as f:
                f.write(buffer)
        else:
            fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_DIRECT, 0o644)
            os.write(fd, buffer)
            os.close(fd)
        disk_write_time = time.time() - start_time
        logger.debug(
            f"Disk write size: {size} bytes, "
            f"Bandwidth: {size / disk_write_time / 1e6:.2f} MB/s"
        )

    @_lmcache_nvtx_annotate
    def read_file(self, key, buffer, path):
        start_time = time.time()
        size = len(buffer)
        fblock_aligned = size % self.os_disk_bs == 0
        if not fblock_aligned and self.use_odirect:
            logger.warning(
                "Cannot use O_DIRECT for this file, "
                "size is not aligned to disk block size."
            )

        try:
            if not fblock_aligned or not self.use_odirect:
                with open(path, "rb") as f:
                    f.readinto(buffer)
            else:
                fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
                with os.fdopen(fd, "rb", buffering=0) as fdo:
                    fdo.readinto(buffer)
        except FileNotFoundError:
            logger.warning(f"File not found on disk: {path}")
            if self.dict.get(key, None):
                self.dict.pop(key)
            return

        disk_read_time = time.time() - start_time
        logger.debug(
            f"Disk read size: {size} bytes, "
            f"Bandwidth: {size / disk_read_time / 1e6:.2f} MB/s"
        )

    def get_allocator_backend(self) -> LocalCPUBackend:
        return self.local_cpu_backend

    def close(self) -> None:
        # Wait for the startup index rebuild so we don't tear down the
        # event loop it's running on while it's still scanning.
        try:
            self._rebuild_index_future.result(timeout=30)
        except Exception as e:
            logger.warning(f"Exception while waiting for index rebuild: {e}")
        if self.batched_msg_sender is not None:
            self.batched_msg_sender.close()
        self._read_thread_pool.shutdown(wait=True)
        self.disk_worker.close()
