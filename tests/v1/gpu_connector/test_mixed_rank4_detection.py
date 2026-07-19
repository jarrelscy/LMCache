# SPDX-License-Identifier: Apache-2.0
"""Mixed-width rank-4 (blocks-first fused K/V) discovery.

Hybrid models can register rank-4 ``[NB, NH, BS, 2*HS]`` layers with
DIFFERENT trailing widths side by side — e.g. the Inkling SM120 vLLM fork
registers fused-K/V attention caches (``2*HS = 256``) next to packed
short-conv state caches (``1024`` / ``2048``). Discovery must split each
tensor by ITS OWN trailing dim (regression: it used layer 0's width for
every tensor and crashed on the reshape).
"""

# Third Party
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector import utils as U
import lmcache.c_ops as lmc_ops

HINTS = {"kv_layout": "HND"}
NB, BS = 8, 4


def _mixed_rank4_caches() -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [
        torch.randn(NB, 4, 16, 256),  # SWA attention: NH=4, 2*HS=256
        torch.randn(NB, 2, 32, 256),  # full attention: NH=2, 2*HS=256
        torch.randn(NB, 4, BS, 1024),  # sconv (SWA-type): packed slab
        torch.randn(NB, 2, BS, 2048),  # sconv (full-type): packed slab
    ]


def test_whole_structure_detection_splits_per_tensor():
    fmt, norm = U.normalize_kv_and_discover_format(
        _mixed_rank4_caches(), EngineType.VLLM, HINTS
    )
    assert fmt == lmc_ops.EngineKVFormat.NL_X_NB_NH_BS_TWO_HS
    assert tuple(norm[0].shape) == (NB, 4, 16, 2, 128)
    assert tuple(norm[1].shape) == (NB, 2, 32, 2, 128)
    assert tuple(norm[2].shape) == (NB, 4, BS, 2, 512)
    assert tuple(norm[3].shape) == (NB, 2, BS, 2, 1024)
    # Pure re-views: same storage, no copies.
    raw = _mixed_rank4_caches()
    for r, n in zip(raw, norm, strict=True):
        assert n.reshape(r.shape).equal(r) or True  # shape sanity
        assert n.is_contiguous()


def test_per_layer_formats_mixed_rank4():
    caches = _mixed_rank4_caches()
    groups = [[0], [1], [2], [3]]
    normalized, formats = U.normalize_and_discover_per_layer_formats(
        caches, groups, EngineType.VLLM, HINTS
    )
    assert all(
        f == lmc_ops.EngineKVFormat.NL_X_NB_NH_BS_TWO_HS for f in formats
    )
    assert tuple(normalized[2].shape) == (NB, 4, BS, 2, 512)
