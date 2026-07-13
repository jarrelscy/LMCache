# SPDX-License-Identifier: Apache-2.0
"""Pure-Python helpers for making LMCache's vLLM adapter aware of vLLM's
Decode Context Parallelism (DCP) token sharding.

Under DCP, vLLM reuses the TP group's GPUs and shards the (MLA/latent) KV
cache across `dcp_world` ranks at `cp_kv_cache_interleave_size`-token
granularity (default interleave=1, i.e. plain round-robin per token). A
"virtual" (shared) block of `block_size * dcp_world` tokens is allocated
as a single block id by the scheduler, but each rank only physically
stores `block_size` of those tokens, at a rank-local, contiguous offset
within its own page.

This module reimplements exactly the addressing formula used by vLLM's
own CUDA/Triton kernel (`_compute_slot_mappings_kernel`,
vllm/v1/worker/gpu/block_table.py, verified against the fork at
/home/jarrelscy/glm52/vllm on 2026-07-13), so a worker computing which of
its LOCAL physical KV slots hold a chunk of GLOBAL token positions gets
bit-identical results to what vLLM itself used when writing that KV
cache.

Deliberately has zero vLLM/torch/torch.distributed imports: this keeps
it trivially unit-testable in isolation (no CUDA, no distributed init
required) and reusable from worker-side adapter code without pulling in
more than plain Python.

Terminology used throughout:
  - "global" token position / chunk: as seen by the scheduler and by
    LMCache's chunk hashing (`token_database.process_tokens`), replicated
    identically on every rank.
  - "local" token position / index: this rank's compacted 0..N-1 view of
    only the tokens it physically owns, in increasing global-position
    order. This is the space LMCache's `hashes=`/`offsets=` store()/
    retrieve() path and the GPU connector's `start`/`end` args operate in
    once hashes/offsets are used instead of tokens/mask.
"""

# Standard
from dataclasses import dataclass, field
from typing import List, Optional


def dcp_is_local_and_local_offset(
    block_offset: int,
    dcp_world: int,
    dcp_rank: int,
    interleave: int,
) -> tuple[bool, int]:
    """Per-position ownership + local-offset formula, mirroring the
    `CP_SIZE > 1` branch of vLLM's `_compute_slot_mappings_kernel`.

    :param block_offset: ``position % (block_size * dcp_world)``, i.e.
        the token's offset within its *virtual* (shared, dcp_world-times
        oversized) block.
    :param dcp_world: number of DCP ranks (``decode_context_parallel_size``).
    :param dcp_rank: this worker's DCP rank (``get_dcp_group().rank_in_group``).
    :param interleave: ``cp_kv_cache_interleave_size`` (vLLM default: 1).
    :returns: ``(is_local, local_offset)``. ``local_offset`` is only
        meaningful when ``is_local`` is True.
    """
    is_local = (block_offset // interleave) % dcp_world == dcp_rank
    rounds = block_offset // (interleave * dcp_world)
    remainder = block_offset % interleave
    local_offset = rounds * interleave + remainder
    return is_local, local_offset


@dataclass
class DcpChunkPlan:
    """This rank's view of one GLOBAL lmcache chunk ``[g_start, g_end)``."""

    g_start: int
    g_end: int
    # Physical slot index per local (this-rank-owned) token, in
    # increasing global-position order. len(local_slots) == local_count.
    local_slots: List[int] = field(default_factory=list)
    # Of local_slots, how many correspond to global positions strictly
    # before `vllm_cached_tokens` (only computed/meaningful when the
    # caller passed a non-None vllm_cached_tokens; see plan_dcp_chunk).
    local_vllm_cached_tokens: int = 0

    @property
    def local_count(self) -> int:
        return len(self.local_slots)


def plan_dcp_chunk(
    g_start: int,
    g_end: int,
    block_id: int,
    block_size: int,
    dcp_world: int,
    dcp_rank: int,
    interleave: int = 1,
    vllm_cached_tokens: Optional[int] = None,
) -> DcpChunkPlan:
    """Compute this rank's local slot_mapping (and, optionally, the local
    equivalent of ``vllm_cached_tokens``) for one GLOBAL lmcache chunk.

    Requires the chunk to fit within exactly one vLLM "virtual" DCP block
    (``g_end - g_start <= block_size * dcp_world``), which holds whenever
    ``lmcache_chunk_size <= block_size * dcp_world_size``. The production
    GLM-5.2 DCP4 profile uses ``lmcache_chunk_size == block_size *
    dcp_world_size`` (256 == 64*4) exactly, so every non-final chunk maps
    1:1 onto one virtual block / one allocated block id; this function
    still works correctly for a shorter final partial chunk.

    :param g_start: global start token index of this chunk (inclusive).
    :param g_end: global end token index of this chunk (exclusive).
    :param block_id: the (shared/virtual) block id covering this chunk,
        i.e. ``tracker.allocated_block_ids[g_start // (block_size * dcp_world)]``.
    :param block_size: vLLM's physical block size (tokens/page/rank).
    :param dcp_world: ``decode_context_parallel_size``.
    :param dcp_rank: this worker's DCP rank.
    :param interleave: ``cp_kv_cache_interleave_size``.
    :param vllm_cached_tokens: if given, also compute
        ``local_vllm_cached_tokens`` = count of this rank's local tokens
        (in increasing global order) whose global position is strictly
        less than ``vllm_cached_tokens``. Used to convert the GLOBAL
        "vLLM already has valid KV for this many leading tokens" count
        into the LOCAL-index-space value the GPU connector's
        ``skip_prefix_n_tokens`` computation needs when start/end are
        local indices (see LMCACHE_FORK_PROGRESS.md).
    """
    virtual_block_size = block_size * dcp_world
    chunk_len = g_end - g_start
    if chunk_len > virtual_block_size:
        raise ValueError(
            f"DCP chunk [{g_start}, {g_end}) spans {chunk_len} tokens, "
            f"more than one virtual DCP block ({virtual_block_size} = "
            f"block_size({block_size}) * dcp_world({dcp_world})). "
            "lmcache_chunk_size must be <= block_size * "
            "decode_context_parallel_size for this profile's addressing "
            "to be valid; a chunk spanning multiple virtual blocks would "
            "need multiple block ids and is not handled here."
        )
    if g_start % virtual_block_size != 0:
        raise ValueError(
            f"DCP chunk global start {g_start} is not aligned to the "
            f"virtual DCP block size ({virtual_block_size}); a single "
            f"block_id ({block_id}) cannot cover this chunk. lmcache "
            "chunk boundaries must be multiples of block_size*dcp_world "
            "under DCP."
        )

    plan = DcpChunkPlan(g_start=g_start, g_end=g_end)
    for p in range(g_start, g_end):
        # Match vLLM's own `positions % (block_size * CP_SIZE)` literally
        # (block_table.py:284) rather than `p - g_start`, so this stays
        # correct even if the alignment check above is ever relaxed.
        block_offset = p % virtual_block_size
        is_local, local_offset = dcp_is_local_and_local_offset(
            block_offset, dcp_world, dcp_rank, interleave
        )
        if not is_local:
            continue
        plan.local_slots.append(block_id * block_size + local_offset)
        if vllm_cached_tokens is not None and p < vllm_cached_tokens:
            plan.local_vllm_cached_tokens += 1
    return plan


def assert_dcp_chunking_compatible(
    lmcache_chunk_size: int, block_size: int, dcp_world: int
) -> None:
    """Fail loudly at connector-init time rather than silently
    miscomputing if the deployment's chunk/block/dcp configuration isn't
    one `plan_dcp_chunk` (and the adapter code built on top of it) has
    been validated for.
    """
    if dcp_world <= 1:
        return
    virtual_block_size = block_size * dcp_world
    if lmcache_chunk_size % virtual_block_size != 0 and (
        virtual_block_size % lmcache_chunk_size != 0
    ):
        raise ValueError(
            f"Under DCP (dcp_world={dcp_world}), lmcache_chunk_size "
            f"({lmcache_chunk_size}) must divide, or be a multiple of, "
            f"the virtual DCP block size (block_size*dcp_world = "
            f"{virtual_block_size}); got neither. Chunk-to-virtual-block "
            "alignment is required for the DCP adapter's per-chunk "
            "addressing to be correct."
        )
    if lmcache_chunk_size > virtual_block_size:
        raise ValueError(
            f"Under DCP (dcp_world={dcp_world}), lmcache_chunk_size "
            f"({lmcache_chunk_size}) > virtual DCP block size "
            f"({virtual_block_size}) is not supported by plan_dcp_chunk "
            "(a chunk spanning >1 block id is not handled)."
        )
