# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for per-block KV coalescing in the Mooncake store worker
(_BlockStager): gather/scatter correctness, wire-format compatibility with the
per-layer multi-buffer path, failed-key isolation, and slot chunking."""

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (
    _BlockStager,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="coalescing staging pool requires CUDA"
)

NUM_BLOCKS = 8
# Heterogeneous per-layer regions (GLM/DeepSeek-DSA-like: two cache shapes).
BLOCK_LENS = [4096, 4096, 4096, 512, 512]


class _FakeStore:
    """In-memory store implementing the three transfer APIs the stager uses.

    Reads/writes the stager's GPU staging pool through the stager's own
    tensor (resolved from the raw addresses it advertises), so the test
    exercises the exact address arithmetic the RDMA engine would use.
    """

    def __init__(self):
        self.data: dict[str, bytes] = {}
        self.registered: list[tuple[int, int]] = []
        self.fail_keys: set[str] = set()
        self.stager: _BlockStager | None = None

    def register_buffer(self, addr: int, size: int) -> int:
        self.registered.append((addr, size))
        return 0

    def _slot_view(self, addr: int) -> torch.Tensor:
        st = self.stager
        assert st is not None
        base = st.staging.data_ptr()
        assert (addr - base) % st.slot_bytes == 0
        return st.staging[(addr - base) // st.slot_bytes]

    def batch_put_from_multi_buffers(self, keys, addrs, sizes, _cfg):
        res = []
        for key, alist, slist in zip(keys, addrs, sizes):
            assert len(alist) == 1 and slist[0] == self.stager.slot_bytes
            self.data[key] = bytes(self._slot_view(alist[0]).cpu().numpy())
            res.append(0)
        return res

    def batch_get_into_multi_buffers(self, keys, addrs, sizes):
        res = []
        for key, alist, slist in zip(keys, addrs, sizes):
            if key in self.fail_keys or key not in self.data:
                res.append(-1)
                continue
            val = self.data[key]
            assert len(alist) == 1 and slist[0] == len(val)
            self._slot_view(alist[0]).copy_(
                torch.frombuffer(bytearray(val), dtype=torch.uint8)
            )
            res.append(len(val))
        return res


def _make_regions(device: str, fill_random: bool = True) -> list[torch.Tensor]:
    torch.manual_seed(7)
    regions = []
    for blk_len in BLOCK_LENS:
        t = torch.empty(NUM_BLOCKS, blk_len, dtype=torch.uint8, device=device)
        if fill_random:
            t.copy_(
                torch.randint(0, 256, (NUM_BLOCKS, blk_len), dtype=torch.uint8)
            )
        else:
            t.zero_()
        regions.append(t)
    return regions


def _make_stager(regions, num_slots=4):
    store = _FakeStore()
    stager = _BlockStager(store, regions, BLOCK_LENS, NUM_BLOCKS, num_slots)
    store.stager = stager
    return store, stager


def _expected_wire_bytes(regions, blk: int) -> bytes:
    """The multi-buffer wire format: per-region fragments concatenated in
    registration order — what an unpatched worker produces/expects."""
    return b"".join(bytes(r[blk].cpu().numpy()) for r in regions)


@requires_cuda
def test_put_get_roundtrip():
    regions = _make_regions("cuda")
    originals = [r.clone() for r in regions]
    store, stager = _make_stager(regions)
    keys = [f"k{i}" for i in range(NUM_BLOCKS)]
    blocks = list(range(NUM_BLOCKS))

    assert stager.put_coalesced(store, keys, blocks, None) == [0] * NUM_BLOCKS
    for r in regions:
        r.zero_()
    res = stager.get_coalesced(store, keys, blocks)
    assert all(v >= 0 for v in res)
    for r, o in zip(regions, originals):
        assert torch.equal(r, o)


@requires_cuda
def test_wire_format_matches_multibuffer_layout():
    regions = _make_regions("cuda")
    store, stager = _make_stager(regions)
    keys = [f"k{i}" for i in range(NUM_BLOCKS)]
    stager.put_coalesced(store, keys, list(range(NUM_BLOCKS)), None)
    for i, key in enumerate(keys):
        assert store.data[key] == _expected_wire_bytes(regions, i)


@requires_cuda
def test_failed_key_never_scatters_stale_bytes():
    regions = _make_regions("cuda")
    store, stager = _make_stager(regions)
    keys = [f"k{i}" for i in range(4)]
    stager.put_coalesced(store, keys, [0, 1, 2, 3], None)

    sentinel = 0xAB
    for r in regions:
        r.fill_(sentinel)
    store.fail_keys = {"k2"}
    res = stager.get_coalesced(store, keys, [0, 1, 2, 3])
    assert res[2] == -1 and all(v >= 0 for i, v in enumerate(res) if i != 2)
    for r in regions:
        # failed key's block rows untouched; others restored
        assert bool((r[2] == sentinel).all())
        assert not bool((r[0] == sentinel).all())


@requires_cuda
def test_chunking_beyond_slot_count():
    regions = _make_regions("cuda")
    originals = [r.clone() for r in regions]
    store, stager = _make_stager(regions, num_slots=2)  # 8 keys -> 4 chunks
    keys = [f"k{i}" for i in range(NUM_BLOCKS)]
    stager.put_coalesced(store, keys, list(range(NUM_BLOCKS)), None)
    for i, key in enumerate(keys):
        assert store.data[key] == _expected_wire_bytes(originals, i)
    for r in regions:
        r.zero_()
    assert all(v >= 0 for v in stager.get_coalesced(store, keys, list(range(NUM_BLOCKS))))
    for r, o in zip(regions, originals):
        assert torch.equal(r, o)


@requires_cuda
def test_matches_rejects_partial_blocks():
    regions = _make_regions("cuda")
    _, stager = _make_stager(regions)
    whole = [list(BLOCK_LENS)]
    partial = [list(BLOCK_LENS[:-1]) + [BLOCK_LENS[-1] // 2]]
    assert stager.matches(whole)
    assert not stager.matches(whole + partial)


@requires_cuda
def test_block_id_mapping_is_positional():
    """Keys map to the block ids passed positionally, not sequentially."""
    regions = _make_regions("cuda")
    store, stager = _make_stager(regions)
    stager.put_coalesced(store, ["a", "b"], [5, 1], None)
    assert store.data["a"] == _expected_wire_bytes(regions, 5)
    assert store.data["b"] == _expected_wire_bytes(regions, 1)


# --- _detect_kv_layout: pure layout classification (no worker / no CUDA) ---
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (  # noqa: E402
    _detect_kv_layout,
)


def test_detect_layout_blocks_first_contiguous_is_coalescable():
    nb = 10
    cache = torch.zeros(nb, 64, dtype=torch.float16)
    region_len = cache.untyped_storage().nbytes()
    addrs, lens, region = _detect_kv_layout(cache, 0x1000, region_len, nb)
    assert addrs == [0x1000]
    assert lens == [region_len // nb]
    assert region is cache  # dense blocks-first -> participates in coalescing


def test_detect_layout_kv_first_splits_and_not_coalescable():
    nb = 10
    cache = torch.zeros(2, nb, 16, 4, 8, dtype=torch.float16)  # (K/V, blocks, ...)
    region_len = cache.untyped_storage().nbytes()
    seg = cache.stride(0) * cache.element_size()
    addrs, lens, region = _detect_kv_layout(cache, 0x2000, region_len, nb)
    assert addrs == [0x2000, 0x2000 + seg]
    assert lens == [seg // nb, seg // nb]
    assert region is None  # split layout -> excluded from coalescing


def test_detect_layout_non_contiguous_blocks_first_excluded():
    nb = 8
    full = torch.zeros(nb, 128, dtype=torch.float16)
    view = full[:, :64]  # blocks-first but strided (not dense) -> not coalescable
    region_len = view.untyped_storage().nbytes()
    addrs, lens, region = _detect_kv_layout(view, 0x3000, region_len, nb)
    assert len(addrs) == 1
    assert region is None
