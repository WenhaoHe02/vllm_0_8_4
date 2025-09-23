import os
import time
import uuid
from collections import defaultdict
from typing import List, Tuple, Optional

import msgspec
import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger

from .kv_rearrange import rearrange_tensors

logger = init_logger(__name__)

# ====== 可选 Triton 支持 ======
_USE_TRITON_REARR = os.getenv("NIXL_USE_TRITON_REARRANGE", "0") == "1"
_USE_TRITON_SUM   = os.getenv("NIXL_USE_TRITON_SUM", "0") == "1"
try:
    if _USE_TRITON_REARR or _USE_TRITON_SUM:
        import triton
        import triton.language as tl
        _TRITON_AVAILABLE = True
    else:
        _TRITON_AVAILABLE = False
except Exception:
    _TRITON_AVAILABLE = False
    _USE_TRITON_REARR = False
    _USE_TRITON_SUM   = False

_WAIT_SPIN    = int(os.getenv("NIXL_WAIT_SPIN", "2000"))
_WAIT_BASE_US = int(os.getenv("NIXL_WAIT_BASE_US", "50"))
_WAIT_MAX_US  = int(os.getenv("NIXL_WAIT_MAX_US", "2000"))
_DEBUG_TIMING = os.getenv("NIXL_DEBUG_TIMING", "0") == "1"

def _cuda_sync_if_debug():
    if _DEBUG_TIMING and torch.cuda.is_available():
        torch.cuda.synchronize()

def _wait_xfer(wrapper, handle):
    spins = 0
    sleep_us = _WAIT_BASE_US
    while True:
        st = wrapper.check_xfer_state(handle)
        if st == "DONE":
            return
        if st != "PROC":
            raise RuntimeError(f"transfer failed: {st}")
        if spins < _WAIT_SPIN:
            spins += 1
            continue
        time.sleep(sleep_us / 1e6)
        sleep_us = min(sleep_us * 2, _WAIT_MAX_US)

def _wait_many(wrapper, handles: List[object]):
    pending = list(handles)
    spins = 0
    sleep_us = _WAIT_BASE_US
    idx = 0
    while pending:
        h = pending[idx]
        st = wrapper.check_xfer_state(h)
        if st == "DONE":
            pending.pop(idx)
            if not pending:
                return
            idx %= len(pending)
            continue
        if st != "PROC":
            raise RuntimeError(f"transfer failed: {st}")
        # 轮转前进
        idx = (idx + 1) % len(pending)
        if idx == 0:
            if spins < _WAIT_SPIN:
                spins += 1
            else:
                time.sleep(sleep_us / 1e6)
                sleep_us = min(sleep_us * 2, _WAIT_MAX_US)

if _TRITON_AVAILABLE and _USE_TRITON_REARR:
    @triton.jit
    def _interleave_heads_kernel(src, dst,
                                 B: tl.constexpr, H: tl.constexpr, D: tl.constexpr):
        t = tl.program_id(0)  # token
        h = tl.program_id(1)  # head
        if h >= H:
            return
        BLOCK_D = 128
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        s = src + t * (H*D) + h * D + offs_d
        d = dst + t * (H*D) + h * D + offs_d
        vals = tl.load(s, mask=mask_d, other=0)
        tl.store(d, vals, mask=mask_d)

    def triton_rearrange(cache_src_3d: torch.Tensor, cache_dst_3d: torch.Tensor):
        assert cache_src_3d.is_cuda and cache_dst_3d.is_cuda
        B, H, D = cache_src_3d.shape
        grid = (B, H)
        _interleave_heads_kernel[grid](
            cache_src_3d, cache_dst_3d, B, H, D,
            num_warps=4, num_stages=2
        )

if _TRITON_AVAILABLE and _USE_TRITON_SUM:
    @triton.jit
    def _sum_int32_kernel(x_ptr, out_ptr, N: tl.constexpr):
        pid = tl.program_id(0)
        BLOCK = 1024
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        x = tl.load(x_ptr + offs, mask=m, other=0).to(tl.int32)
        s = tl.sum(x, axis=0)
        if tl.arange(0, 1) == 0:
            tl.atomic_add(out_ptr, s)

    def triton_sum_int32(x: torch.Tensor) -> int:
        assert x.is_cuda
        N = x.numel()
        out = torch.zeros((), dtype=torch.int64, device=x.device)
        grid = ((N + 1023) // 1024,)
        _sum_int32_kernel[grid](x, out, N, num_warps=4)
        return int(out.item())

_CACHE_DESC = os.getenv("NIXL_CACHE_DESC", "1") == "1"

try:
    from nixl._api import nixl_agent as NixlWrapper
    logger.info("NIXL is available")
except ImportError:
    logger.warning("NIXL is not available")
    NixlWrapper = None


class NixlMetadata(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    dict=True,
):
    engine_id: str
    agent_metadata: List[bytes]
    kv_caches_base_addr: List[List[List[int]]]
    num_blocks: int
    kv_caches_dev_ids: Optional[List[List[List[int]]]] = None


class DynamoNixlConnector:
    def __init__(self, vllm_config: VllmConfig, engine_id: str, rank: int):
        self.vllm_config = vllm_config
        if NixlWrapper is None:
            logger.error("NIXL is not available")
            raise RuntimeError("NIXL is not available")
        logger.info("Initializing NIXL wrapper")
        self.nixl_wrapper = NixlWrapper(str(uuid.uuid4()), None)

        self.use_prepped_xfer = vllm_config.kv_transfer_config.use_prepped_xfer

        self.num_layers = None
        self.num_blocks = None
        self.num_heads = None
        self.block_len = None
        self.kv_caches = None
        self.kv_caches_base_addr = {}
        self.kv_cache_shape = {}

        self._registered_descs = []
        self._remote_agents = {}
        self.engine_id = engine_id
        self.rank = rank
        self._tp_size = {}
        self.src_xfer_side_handles = {}
        self.prefill_dst_xfer_side_handles = defaultdict(dict)
        self.dst_xfer_side_handles = defaultdict(dict)
        self.dst_num_blocks = {}

        self.kv_caches_dev_ids = {}

        self._transfers = defaultdict(list)

        self._tp_size[engine_id] = vllm_config.parallel_config.tensor_parallel_size
        self._is_mla = "deepseek" in vllm_config.model_config.architectures[0].lower()

        self.block_size = None
        self.head_dim = None

        self._downscale_info = {}

        # desc 缓存（token 区间）
        self._desc_cache = {
            "local_token_desc": {},   # key: tuple((lo,hi),...)-> List[int]
            "remote_token_desc": {},  # key: (engine_id, tuple((lo,hi),...))-> List[int]
        }

    def _expand_blocks_to_tokens(self, block_ids: List[int]) -> List[int]:
        B = int(self.block_size)
        out = []
        base_cache = {}
        for b in block_ids:
            base = base_cache.get(b)
            if base is None:
                base = b * B
                base_cache[b] = base
            out.extend(range(base, base + B))
        return out

    def _token_ranges_from_blocks(self, block_ids: List[int]) -> List[tuple]:
        # 把 block 连续段映射为 token 连续区间，减少 per-token append
        B = int(self.block_size)
        if not block_ids:
            return []
        ids = sorted(block_ids)
        ranges = []
        start_b = prev_b = ids[0]
        for b in ids[1:]:
            if b == prev_b + 1:
                prev_b = b
                continue
            # 结束上一段
            s_tok = start_b * B
            e_tok = (prev_b + 1) * B - 1
            ranges.append((s_tok, e_tok))
            start_b = prev_b = b
        s_tok = start_b * B
        e_tok = (prev_b + 1) * B - 1
        ranges.append((s_tok, e_tok))
        return ranges

    def _local_token_desc_ids_by_ranges(self, token_ranges: List[tuple]) -> List[int]:
        if _CACHE_DESC:
            key = tuple(token_ranges)
            c = self._desc_cache["local_token_desc"].get(key)
            if c is not None:
                return c
        per_entry = self.num_blocks * self.block_size
        out = []
        for layer_id in range(self.num_layers):
            base_L = layer_id * self.num_cache_entries * per_entry
            for entry_index in range(self.num_cache_entries):
                base_LE = base_L + entry_index * per_entry
                for lo, hi in token_ranges:
                    out.extend(range(base_LE + lo, base_LE + hi + 1))
        if _CACHE_DESC:
            self._desc_cache["local_token_desc"][key] = out
        return out

    def _remote_token_desc_ids_by_ranges(self, engine_id: str, token_ranges: List[tuple]) -> List[int]:
        if _CACHE_DESC:
            key = (engine_id, tuple(token_ranges))
            c = self._desc_cache["remote_token_desc"].get(key)
            if c is not None:
                return c
        num_units = self.dst_num_blocks[engine_id]  # DOWN: 单位=token；UP/EQ: 单位=block
        out = []
        for layer_id in range(self.num_layers):
            base_L = layer_id * self.num_cache_entries * num_units
            for entry_index in range(self.num_cache_entries):
                base_LE = base_L + entry_index * num_units
                for lo, hi in token_ranges:
                    out.extend(range(base_LE + lo, base_LE + hi + 1))
        if _CACHE_DESC:
            self._desc_cache["remote_token_desc"][key] = out
        return out

    # ============ DOWN 写/读 ============
    def _write_blocks_down(self, local_block_ids, remote_block_ids, dst_engine_id, notify_msg):
        down = self._downscale_info[dst_engine_id]
        assert down is not None, "[WRITE-DOWN] downscale info missing"

        loc_tok_ranges = self._token_ranges_from_blocks(local_block_ids)
        rem_tok_ranges = self._token_ranges_from_blocks(remote_block_ids)

        local_desc_ids  = self._local_token_desc_ids_by_ranges(loc_tok_ranges)
        remote_desc_ids = self._remote_token_desc_ids_by_ranges(dst_engine_id, rem_tok_ranges)

        local_handle  = self.src_xfer_side_handles[1]  # token 粒度句柄
        remote_handle = self.dst_xfer_side_handles[dst_engine_id][0]

        h = self.nixl_wrapper.make_prepped_xfer(
            "WRITE",
            local_handle, local_desc_ids,
            remote_handle, remote_desc_ids,
            ""  # 不带 payload
        )
        self.nixl_wrapper.transfer(h)
        _wait_xfer(self.nixl_wrapper, h)

        payload = notify_msg if isinstance(notify_msg, str) else str(notify_msg)
        t0 = time.perf_counter()
        self._barrier_mark_and_wait(
            dst_engine_id,
            payload,
            down["group_size"],
            down["peer_idx"],
            down["notify_leader"],
        )
        if down["notify_leader"]:
            trg = self._remote_agents[dst_engine_id][down["remote_rank"]]
            self.nixl_wrapper.send_notif(trg, payload)
        if _DEBUG_TIMING:
            logger.info("[WRITE-DOWN] barrier+notify_ms=%.3f", (time.perf_counter() - t0) * 1000)

    def _read_blocks_down(self, local_block_ids, remote_block_ids, dst_engine_id):
        down = self._downscale_info[dst_engine_id]
        assert down is not None, "[READ-DOWN] downscale info missing"

        loc_tok_ranges = self._token_ranges_from_blocks(local_block_ids)
        rem_tok_ranges = self._token_ranges_from_blocks(remote_block_ids)

        local_handle  = self.src_xfer_side_handles[1]
        remote_handle = self.dst_xfer_side_handles[dst_engine_id][0]
        staging_desc_ids = self._local_token_desc_ids_by_ranges(loc_tok_ranges)
        remote_desc_ids  = self._remote_token_desc_ids_by_ranges(dst_engine_id, rem_tok_ranges)

        h = self.nixl_wrapper.make_prepped_xfer(
            "READ",
            local_handle, staging_desc_ids,
            remote_handle, remote_desc_ids,
            ""
        )
        self.nixl_wrapper.transfer(h)
        _wait_xfer(self.nixl_wrapper, h)

    # ============ 其它工具 ============
    def _local_token_desc_ids(self, token_ids: List[int]) -> List[int]:
        # 兼容老路径；新路径已改区间/缓存
        per_entry = self.num_blocks * self.block_size
        ids = []
        for layer_id in range(self.num_layers):
            for entry_index in range(self.num_cache_entries):
                for tok_id in token_ids:
                    ids.append(layer_id * self.num_cache_entries * per_entry +
                               entry_index * per_entry + tok_id)
        return ids

    def _kv_block_u32sum(self, layer: int, entry_idx: int, block_id: int) -> int:
        t = self.kv_caches[layer][entry_idx][block_id]  # [B, H_local, D]
        if _TRITON_AVAILABLE and _USE_TRITON_SUM and t.is_cuda:
            return triton_sum_int32(t)
        return int(t.view(torch.int32).sum().item())

    def _down_verify_peer_segment(
        self,
        dst_engine_id: str,
        remote_block_id: int,
        scratch_block_id: Optional[int] = None,
        max_layers: int = 2,
    ) -> None:
        if scratch_block_id is None:
            scratch_block_id = (remote_block_id + 1) % max(1, self.num_blocks)

        self.read_blocks(
            local_block_ids=[scratch_block_id],
            staging_block_ids=[scratch_block_id],
            remote_block_ids=[remote_block_id],
            dst_engine_id=dst_engine_id,
        )

        L = min(max_layers, self.num_layers)
        k_ok = True
        v_ok = True
        for layer in range(L):
            src_k = self._kv_block_u32sum(layer, 0, remote_block_id)
            dst_k = self._kv_block_u32sum(layer, 0, scratch_block_id)
            src_v = self._kv_block_u32sum(layer, 1, remote_block_id)
            dst_v = self._kv_block_u32sum(layer, 1, scratch_block_id)
            k_ok = k_ok and (src_k == dst_k)
            v_ok = v_ok and (src_v == dst_v)
            logger.info(
                "[DOWN-CHK] engine=%s layer=%d block=%d -> scratch=%d K:%d==%d %s V:%d==%d %s",
                dst_engine_id, layer, remote_block_id, scratch_block_id,
                src_k, dst_k, "OK" if src_k == dst_k else "MISMATCH",
                src_v, dst_v, "OK" if src_v == dst_v else "MISMATCH",
            )
        logger.info(
            "[DOWN-CHK] summary: K=%s V=%s (layers checked=%d)",
            "OK" if k_ok else "MISMATCH",
            "OK" if v_ok else "MISMATCH",
            L,
        )

    def _down_peer_perm(self, group_size: int):
        s = os.getenv("NIXL_DOWN_ORDER", "").strip()
        if not s:
            return list(range(group_size))
        try:
            parts = [int(x) for x in s.split(",")]
            if sorted(parts) == list(range(group_size)):
                return parts
        except Exception:
            pass
        logger.warning("[DOWN-PERM] invalid NIXL_DOWN_ORDER=%r, fallback identity", s)
        return list(range(group_size))

    def _sanitize_key(self, s: object, maxlen: int = 40) -> str:
        from uuid import uuid4
        s = str(s)
        out = []
        for ch in s:
            if ch.isalnum() or ch in ("-", "_"):
                out.append(ch)
            if len(out) >= maxlen:
                break
        return "".join(out) or uuid4().hex[:maxlen]

    def _barrier_dir(self, dst_engine_id: str, notify_key: str, group_size: int) -> str:
        base = os.getenv("NIXL_BARRIER_DIR", "/dev/shm" if os.path.isdir("/dev/shm") else "/tmp")
        safe_engine = self._sanitize_key(dst_engine_id, 16)
        safe_key = self._sanitize_key(notify_key, 24)
        d = os.path.join(base, f"nixl_down_bar_{safe_engine}_{safe_key}_{group_size}")
        os.makedirs(d, exist_ok=True)
        return d

    def _barrier_mark_and_wait(self, dst_engine_id: str, notify_key: str,
                               group_size: int, peer_idx: int, is_leader: bool) -> None:
        d = self._barrier_dir(dst_engine_id, notify_key, group_size)
        my_flag = os.path.join(d, f"{peer_idx}.ok")
        try:
            with open(my_flag, "w") as f:
                f.write("ok")
        except Exception as e:
            logger.warning("[DOWN-BAR] write flag failed: %s", e)

        if not is_leader:
            return
        try:
            for i in range(group_size):
                flag = os.path.join(d, f"{i}.ok")
                while not os.path.exists(flag):
                    time.sleep(0.0005)  # 更细粒度轮询，配合自旋等待
            # 清理
            for i in range(group_size):
                try:
                    os.remove(os.path.join(d, f"{i}.ok"))
                except OSError:
                    pass
            try:
                os.rmdir(d)
            except OSError:
                pass
        except Exception as e:
            logger.warning("[DOWN-BAR] wait/cleanup failed: %s", e)

    @property
    def agent_name(self):
        return self.nixl_wrapper.name

    # ============ 注册 ============

    def register_kv_caches(self, kv_caches: List[torch.Tensor]):
        logger.debug("--------------------------------")
        logger.debug("Registering kv caches for engine %s", self.engine_id)
        logger.debug(f"Is deepseek: {self._is_mla}")
        logger.debug(f"kv_cache shape: {kv_caches[0].shape}")
        logger.debug("--------------------------------")

        if self._is_mla:
            num_blocks, block_size, head_dim = kv_caches[0].shape
            self.block_len = head_dim * block_size * kv_caches[0].element_size()
            self.num_layers = len(kv_caches)
            self.num_blocks = num_blocks
            self.num_heads = 1
            self.kv_caches = kv_caches
            self.num_cache_entries = 1

            kv_caches_base_addr = []
            caches_data = []
            for kv_cache in kv_caches:
                base_addr = kv_cache.data_ptr()
                region_len = self.num_cache_entries * num_blocks * self.block_len
                caches_data.append((base_addr, region_len, self.rank, ""))
                kv_caches_base_addr.append([base_addr, ])

            self.kv_caches_base_addr[self.engine_id] = kv_caches_base_addr

            descs = self.nixl_wrapper.get_reg_descs(caches_data, "VRAM")
            self.nixl_wrapper.register_memory(descs)
            self._registered_descs.append(descs)
        else:
            _, num_blocks, block_size, num_heads, head_dim = kv_caches[0].shape
            self.block_len = block_size * num_heads * head_dim * kv_caches[0].element_size()
            self.block_size = block_size
            self.head_dim = head_dim
            self.num_layers = len(kv_caches)
            self.num_blocks = num_blocks
            self.num_heads = num_heads
            self.kv_caches = kv_caches
            self.num_cache_entries = 2

            kv_caches_base_addr = []
            caches_data = []
            for key_cache, value_cache in kv_caches:
                base_addr = key_cache.data_ptr()
                region_len = self.num_cache_entries * num_blocks * self.block_len
                caches_data.append((base_addr, region_len, self.rank, ""))
                kv_caches_base_addr.append([key_cache.data_ptr(), value_cache.data_ptr()])

            self.kv_caches_base_addr[self.engine_id] = kv_caches_base_addr

            descs = self.nixl_wrapper.get_reg_descs(caches_data, "VRAM")
            self.nixl_wrapper.register_memory(descs)
            self._registered_descs.append(descs)
            logger.info(
                "[KVREG] engine=%s layers=%d blocks=%d entries=%d block_len=%dB elem=%d heads=%s head_dim=%s block_size=%s",
                self.engine_id, self.num_layers, self.num_blocks, self.num_cache_entries,
                self.block_len, kv_caches[0].element_size(), self.num_heads, self.head_dim, self.block_size
            )

    def get_agent_metadata(self):
        return self.nixl_wrapper.get_agent_metadata()

    def shutdown(self):
        for descs_list in self._registered_descs:
            self.nixl_wrapper.deregister_memory(descs_list)
        for agent_names in self._remote_agents.values():
            for agent_name in agent_names:
                self.nixl_wrapper.remove_remote_agent(agent_name)
        for src_xfer_side_handle in self.src_xfer_side_handles.values():
            self.nixl_wrapper.release_dlist_handle(src_xfer_side_handle)
        for dst_xfer_side_handles in self.dst_xfer_side_handles.values():
            for dst_xfer_side_handle in dst_xfer_side_handles.values():
                self.nixl_wrapper.release_dlist_handle(dst_xfer_side_handle)
        for prefill_dst_xfer_side_handles in self.prefill_dst_xfer_side_handles.values():
            for prefill_dst_xfer_side_handle in prefill_dst_xfer_side_handles.values():
                self.nixl_wrapper.release_dlist_handle(prefill_dst_xfer_side_handle)

    # ============ desc 索引 / 范围工具 ============

    def _get_ranges(self, block_ids: List[int]):
        ranges = []
        for i in range(len(block_ids)):
            if i == 0 or block_ids[i] != block_ids[i - 1] + 1:
                ranges.append([block_ids[i], block_ids[i]])
            else:
                ranges[-1][1] = block_ids[i]
        return ranges

    def _get_block_descs_ids(self, engine_id, layer_ids, block_ids, i=None, tp_multiplier=1, staging_ranges=None):
        if layer_ids == "all":
            layer_ids = list(range(self.num_layers))
        if block_ids == "all":
            block_ids = list(range(self.num_blocks))

        descs_ids = []

        if i is not None:
            num_blocks = self.num_blocks
            for layer_id in layer_ids:
                for entry_index in range(self.num_cache_entries):
                    staging_range_idx = 0
                    for block_id in block_ids:
                        if staging_ranges is not None:
                            while staging_range_idx < len(staging_ranges) and (
                                block_id > staging_ranges[staging_range_idx][1]
                                or block_id < staging_ranges[staging_range_idx][0]
                            ):
                                staging_range_idx += 1
                            if staging_range_idx >= len(staging_ranges):
                                raise IndexError("[DESC] staging_range_idx OOB")
                            start_offset = staging_ranges[staging_range_idx][0]
                            i_offset = i * (staging_ranges[staging_range_idx][-1] - start_offset + 1)
                            descs_ids.append(
                                layer_id * self.num_cache_entries * num_blocks * tp_multiplier
                                + entry_index * num_blocks * tp_multiplier
                                + start_offset * tp_multiplier
                                + i_offset + (block_id - start_offset)
                            )
                        else:
                            descs_ids.append(
                                layer_id * self.num_cache_entries * num_blocks
                                + entry_index * num_blocks + block_id
                            )
        else:
            num_blocks = self.dst_num_blocks[engine_id]
            for layer_id in layer_ids:
                for entry_index in range(self.num_cache_entries):
                    for block_id in block_ids:
                        descs_ids.append(
                            layer_id * self.num_cache_entries * num_blocks
                            + entry_index * num_blocks + block_id
                        )
        return descs_ids

    def _get_same_length_ranges(self, src_ranges, dst_ranges, return_original_src_ranges=False):
        src_overlapping_ranges, dst_overlapping_ranges = [], []
        original_src_ranges = []
        org_src_range = tuple(src_ranges[0])

        src_idx, dst_idx = 0, 0
        while src_idx < len(src_ranges) and dst_idx < len(dst_ranges):
            src_range = src_ranges[src_idx]
            dst_range = dst_ranges[dst_idx]

            src_len = src_range[-1] - src_range[0] + 1
            dst_len = dst_range[-1] - dst_range[0] + 1

            if src_len == dst_len:
                src_overlapping_ranges.append([src_range[0], src_range[-1]])
                dst_overlapping_ranges.append([dst_range[0], dst_range[-1]])
                original_src_ranges.append(org_src_range)
                src_idx += 1
                dst_idx += 1
                if src_idx < len(src_ranges):
                    org_src_range = tuple(src_ranges[src_idx])
            elif src_len > dst_len:
                src_overlapping_ranges.append([src_range[0], src_range[0] + dst_len - 1])
                dst_overlapping_ranges.append([dst_range[0], dst_range[-1]])
                original_src_ranges.append(org_src_range)
                src_ranges[src_idx] = [src_range[0] + dst_len, src_range[-1]]
                dst_idx += 1
            else:
                src_overlapping_ranges.append([src_range[0], src_range[-1]])
                dst_overlapping_ranges.append([dst_range[0], dst_range[0] + src_len - 1])
                original_src_ranges.append(org_src_range)
                dst_ranges[dst_idx] = [dst_range[0] + src_len, dst_range[-1]]
                src_idx += 1
                if src_idx < len(src_ranges):
                    org_src_range = tuple(src_ranges[src_idx])
        if return_original_src_ranges:
            return src_overlapping_ranges, dst_overlapping_ranges, original_src_ranges
        return src_overlapping_ranges, dst_overlapping_ranges

    @staticmethod
    def _peek(xs, k=3):
        return xs[:k] + (["..."] if len(xs) > k else [])

    # ============ READ / WRITE 主流程 ============

    def read_blocks(self, local_block_ids, staging_block_ids, remote_block_ids, dst_engine_id):
        logger.info("[READ] local=%s staging=%s remote=%s dst_engine=%s",
                    len(local_block_ids), len(staging_block_ids), len(remote_block_ids), dst_engine_id)
        assert len(local_block_ids) == len(staging_block_ids) == len(remote_block_ids), \
            f"[READ] len mismatch: local={len(local_block_ids)} staging={len(staging_block_ids)} remote={len(remote_block_ids)}"
        if len(local_block_ids) == 0:
            logger.info("[READ] no-op (0 blocks)")
            return

        start_time = time.perf_counter()
        _cuda_sync_if_debug()

        downscale_info = self._downscale_info.get(dst_engine_id)
        tp_multiplier = self._tp_size[dst_engine_id] // self._tp_size[self.engine_id]
        if downscale_info is not None:
            self._read_blocks_down(local_block_ids, remote_block_ids, dst_engine_id)
            logger.info("[READ] total_ms=%.3f (DOWN)",
                        (time.perf_counter() - start_time) * 1000.0)
            return

        # UP/EQ：可能需要重排
        if self._is_mla:
            staging_rearranging_ranges = None
            staging_block_ids = local_block_ids
        else:
            local_ranges = self._get_ranges(local_block_ids)
            staging_ranges = self._get_ranges(staging_block_ids)
            local_rearranging_ranges, staging_rearranging_ranges = self._get_same_length_ranges(local_ranges,
                                                                                                staging_ranges)

        eff_tp = max(1, tp_multiplier)
        targets = list(range(eff_tp))
        remote_block_descs_ids = self._get_block_descs_ids(dst_engine_id, "all", remote_block_ids)
        local_xfer_side_handle = self.src_xfer_side_handles[eff_tp]
        if dst_engine_id not in self.dst_xfer_side_handles:
            raise RuntimeError(f"[READ] dst_xfer_side_handles missing for engine {dst_engine_id}")

        handles = []
        t0 = time.perf_counter()
        for i in targets:
            staging_block_descs_ids = self._get_block_descs_ids(
                self.engine_id, "all", staging_block_ids, i=i, tp_multiplier=eff_tp,
                staging_ranges=staging_rearranging_ranges
            )
            remote_xfer_side_handle = self.dst_xfer_side_handles[dst_engine_id][i]
            h = self.nixl_wrapper.make_prepped_xfer(
                "READ",
                local_xfer_side_handle, staging_block_descs_ids,
                remote_xfer_side_handle, remote_block_descs_ids,
                ""
            )
            self.nixl_wrapper.transfer(h)
            handles.append(h)
        logger.info("[READ] created_transfers=%s create_ms=%.3f",
                    len(handles), (time.perf_counter() - t0) * 1000.0)

        t1 = time.perf_counter()
        _wait_many(self.nixl_wrapper, handles)
        logger.info("[READ] transfer_ms=%.3f", (time.perf_counter() - t1) * 1000.0)

        t2 = time.perf_counter()
        if not self._is_mla:
            for local_range, staging_range in zip(local_rearranging_ranges, staging_rearranging_ranges):
                for kv_cache in self.kv_caches:
                    for cache in kv_cache:
                        # 取出 [B, H, D] 视图
                        src = cache[staging_range[0]:staging_range[1] + 1]  # [B,H,D]
                        dst = cache[local_range[0]:local_range[1] + 1]
                        if _TRITON_AVAILABLE and _USE_TRITON_REARR and src.is_cuda and dst.is_cuda:
                            triton_rearrange(src, dst)
                        else:
                            rearrange_tensors(dst, src, eff_tp, "read")
        logger.info("[READ] rearrange_ms=%.3f total_ms=%.3f",
                    (time.perf_counter() - t2) * 1000.0,
                    (time.perf_counter() - start_time) * 1000.0)

    def write_blocks(self, local_block_ids, staging_block_ids, remote_block_ids, dst_engine_id, notify_msg):
        try:
            logger.info("[WRITE] begin dst=%s local=%d staging=%d remote=%d notify_type=%s",
                        dst_engine_id, len(local_block_ids), len(staging_block_ids),
                        len(remote_block_ids), type(notify_msg).__name__)

            assert len(staging_block_ids) == len(local_block_ids), \
                f"[WRITE] len mismatch: staging={len(staging_block_ids)} local={len(local_block_ids)}"
            assert len(remote_block_ids) == len(local_block_ids), \
                f"[WRITE] len mismatch: remote={len(remote_block_ids)} local={len(local_block_ids)}"

            down = self._downscale_info.get(dst_engine_id)
            tp_multiplier = self._tp_size[dst_engine_id] // self._tp_size[self.engine_id]
            _cuda_sync_if_debug()

            def _to_notify_str(x):
                return x if isinstance(x, str) else str(x)

            if down is not None:
                self._write_blocks_down(local_block_ids, remote_block_ids, dst_engine_id, notify_msg)
                if os.getenv("NIXL_DOWN_VERIFY", "0") == "1":
                    try:
                        if remote_block_ids:
                            self._down_verify_peer_segment(dst_engine_id, remote_block_ids[0])
                    except Exception as e:
                        logger.warning("[DOWN-CHK] verify failed: %s", e)
                logger.info("[WRITE] end ok dst=%s (DOWN)", dst_engine_id)
                return

            eff_tp = max(1, tp_multiplier)
            targets = list(range(eff_tp))

            do_rearrange = False
            staging_rearranging_ranges = None
            if not self._is_mla:
                local_ranges = self._get_ranges(local_block_ids)
                staging_ranges = self._get_ranges(staging_block_ids)
                _local_rearranging_ranges, staging_rearranging_ranges = self._get_same_length_ranges(
                    local_ranges, staging_ranges
                )
                do_rearrange = True

            if not local_block_ids:
                logger.info("[WRITE] zero-block case")
                for i in range(tp_multiplier):
                    trg = self._remote_agents[dst_engine_id][self.rank * tp_multiplier + i]
                    self.nixl_wrapper.send_notif(trg, _to_notify_str(notify_msg))
                logger.info("[WRITE] zero-block broadcast sent (tp=%s)", tp_multiplier)
                return

            if do_rearrange:
                t0 = time.perf_counter()
                for l_rng, s_rng in zip(_local_rearranging_ranges, staging_rearranging_ranges):
                    for kv_cache in self.kv_caches:
                        for cache in kv_cache:
                            src = cache[s_rng[0]: s_rng[1] + 1]
                            dst = cache[l_rng[0]: l_rng[1] + 1]
                            if _TRITON_AVAILABLE and _USE_TRITON_REARR and src.is_cuda and dst.is_cuda:
                                triton_rearrange(src, dst)
                            else:
                                rearrange_tensors(dst, src, eff_tp, "write")
                logger.info("[WRITE] rearrange_ms=%.3f", (time.perf_counter() - t0) * 1000)

            remote_block_descs_ids = self._get_block_descs_ids(dst_engine_id, "all", remote_block_ids)
            local_handle = self.src_xfer_side_handles[eff_tp]
            handles = []

            notify_payload_str = _to_notify_str(notify_msg)

            for i in targets:
                staging_block_descs_ids = self._get_block_descs_ids(
                    self.engine_id, "all", staging_block_ids,
                    i=i, tp_multiplier=eff_tp, staging_ranges=staging_rearranging_ranges
                )
                remote_handle = self.dst_xfer_side_handles[dst_engine_id][i]
                h = self.nixl_wrapper.make_prepped_xfer(
                    "WRITE",
                    local_handle, staging_block_descs_ids,
                    remote_handle, remote_block_descs_ids,
                    notify_payload_str  # UP/EQ：允许 sideband 一起带
                )
                if notify_payload_str:
                    self._transfers.setdefault(notify_payload_str, []).append(h)
                self.nixl_wrapper.transfer(h)
                handles.append(h)

            t1 = time.perf_counter()
            _wait_many(self.nixl_wrapper, handles)
            logger.info("[WRITE] local_xfer_wait_ms=%.3f", (time.perf_counter() - t1) * 1000)
            logger.info("[WRITE] end ok dst=%s (UP/EQ)", dst_engine_id)

        except Exception as e:
            try:
                logger.error(
                    "[WRITE] exception dst=%s down=%s tp_src=%s tp_dst=%s tp_mult=%s rank=%s "
                    "local=%s staging=%s remote=%s notify_repr=%r",
                    dst_engine_id, bool(self._downscale_info.get(dst_engine_id)),
                    self._tp_size.get(self.engine_id), self._tp_size.get(dst_engine_id),
                    (self._tp_size.get(dst_engine_id, 0) // max(1, self._tp_size.get(self.engine_id, 1))),
                    self.rank, len(local_block_ids), len(staging_block_ids), len(remote_block_ids),
                    notify_msg
                )
            finally:
                raise

    # ============ notifs / add_remote_agent / done ============

    def get_notifs(self):
        notifs = self.nixl_wrapper.update_notifs()
        if notifs:
            logger.info("[NOTIF] update_notifs count=%d sample=%s", len(notifs), self._peek(notifs, 4))
        else:
            logger.debug("[NOTIF] update_notifs empty")
        return notifs

    def get_new_notifs(self):
        return self.nixl_wrapper.get_new_notifs()

    def add_remote_agent(
        self,
        engine_id: str,
        agent_metadata: List[bytes],
        agent_tp: int,
        kv_caches_base_addr: List[List[List[int]]],
        num_blocks: int,
        kv_caches_dev_ids: Optional[List[List[List[int]]]] = None,
    ):
        logger.info("[ADD] num_blocks=%d dev_ids=%s", num_blocks, "Y" if kv_caches_dev_ids is not None else "N")
        logger.info("[ADD] engine=%s local_rank=%s local_tp=%s agent_tp=%s is_mla=%s",
                    engine_id, self.rank, self._tp_size[self.engine_id], agent_tp, self._is_mla)

        self._tp_size[engine_id] = int(agent_tp)

        agent_names: List[str] = []
        for meta in agent_metadata:
            agent_names.append(self.nixl_wrapper.add_remote_agent(meta))
        self._remote_agents[engine_id] = agent_names

        self.kv_caches_base_addr[engine_id] = kv_caches_base_addr
        self.kv_caches_dev_ids[engine_id] = kv_caches_dev_ids if kv_caches_dev_ids is not None else None
        loc_base = self.kv_caches_base_addr[engine_id]
        loc_dev = self.kv_caches_dev_ids[engine_id]

        assert len(agent_metadata) == agent_tp
        assert len(loc_base) == agent_tp
        for r in range(agent_tp):
            assert len(loc_base[r]) == self.num_layers
            for L in range(self.num_layers):
                assert len(loc_base[r][L]) == self.num_cache_entries

        tp_multiplier = self._tp_size[engine_id] // self._tp_size[self.engine_id]
        logger.info("[ADD] tp_multiplier=%s (dst_tp/src_tp = %s/%s)",
                    tp_multiplier, self._tp_size[engine_id], self._tp_size[self.engine_id])

        if tp_multiplier == 0 and not self._is_mla:
            group_size = self._tp_size[self.engine_id] // max(1, self._tp_size[engine_id])
            remote_rank = self.rank // group_size
            peer_idx = self.rank % group_size
            slot = peer_idx

            B = int(self.block_size)
            token_len_local = self.block_len // B
            token_len_total = token_len_local * group_size
            seg_len = token_len_local * B
            full_len = token_len_total * B
            peer_off_tok = slot * token_len_local

            self._downscale_info[engine_id] = {
                "group_size": group_size,
                "remote_rank": remote_rank,
                "peer_idx": peer_idx,
                "notify_leader": (peer_idx == 0),
                "perm": None,
                "token_granularity": True,
            }

            self.dst_num_blocks[engine_id] = num_blocks * B

            if 1 not in self.src_xfer_side_handles:
                self.src_xfer_side_handles[1] = None
            local_dev_id = int(torch.cuda.current_device())
            src_blocks = []
            for layer in range(self.num_layers):
                for base in self.kv_caches_base_addr[self.engine_id][layer]:  # K、V
                    for bid in range(self.num_blocks):
                        base_block = base + bid * seg_len
                        for t in range(B):
                            src_blocks.append((base_block + t * token_len_local, token_len_local, local_dev_id))
            src_desc = self.nixl_wrapper.get_xfer_descs(src_blocks, "VRAM")
            self.src_xfer_side_handles[1] = self.nixl_wrapper.prep_xfer_dlist("", src_desc)

            if engine_id not in self.dst_xfer_side_handles:
                self.dst_xfer_side_handles[engine_id] = {}
            dst_blocks = []
            remote_dev_table = self.kv_caches_dev_ids.get(engine_id)
            for layer in range(self.num_layers):
                layer_bases = self.kv_caches_base_addr[engine_id][remote_rank][layer]
                layer_dev_ids = None
                if remote_dev_table is not None:
                    layer_dev_ids = remote_dev_table[remote_rank][layer]
                for entry_idx, rbase in enumerate(layer_bases):  # K、V
                    rdev = (layer_dev_ids[entry_idx] if layer_dev_ids is not None else int(remote_rank))
                    for bid in range(num_blocks):
                        base_block = rbase + bid * full_len
                        for t in range(B):
                            dst_blocks.append((base_block + t * token_len_total + peer_off_tok,
                                               token_len_local, int(rdev)))
            dst_desc = self.nixl_wrapper.get_xfer_descs(dst_blocks, "VRAM")
            self.dst_xfer_side_handles[engine_id][0] = self.nixl_wrapper.prep_xfer_dlist(
                self._remote_agents[engine_id][remote_rank], dst_desc
            )

            # 连接预热（可开关）
            if os.getenv("NIXL_CONNECT_WARMUP", "0") == "1":
                try:
                    self.nixl_wrapper.make_connection(self._remote_agents[engine_id][remote_rank])
                except Exception as e:
                    logger.debug("make_connection(%s) lazy: %s", self._remote_agents[engine_id][remote_rank], e)

            # 为简化后续 id 计算：把对端 TP 伪装成本端
            self._tp_size[engine_id] = self._tp_size[self.engine_id]

            logger.info("[ADD] downscale prepared: src_keys=%s dst_keys=%s dst_units(token)=%s",
                        list(self.src_xfer_side_handles.keys()),
                        list(self.dst_xfer_side_handles[engine_id].keys()),
                        self.dst_num_blocks[engine_id])
            return agent_names

        assert tp_multiplier > 0, f"[ADD] invalid tp_multiplier={tp_multiplier}"
        dst_block_len = self.block_len if self._is_mla else (self.block_len // tp_multiplier)

        if tp_multiplier not in self.src_xfer_side_handles:
            blocks_data = []
            for layer_id in range(self.num_layers):
                for base_addr in self.kv_caches_base_addr[self.engine_id][layer_id]:
                    for block_id in range(self.num_blocks):
                        block_offset = block_id * self.block_len
                        for i in range(1 if self._is_mla else tp_multiplier):
                            tp_off = i * dst_block_len
                            blocks_data.append((base_addr + block_offset + tp_off, dst_block_len, self.rank))
            descs = self.nixl_wrapper.get_xfer_descs(blocks_data, "VRAM")
            self.src_xfer_side_handles[tp_multiplier] = self.nixl_wrapper.prep_xfer_dlist("", descs)

        self.dst_num_blocks[engine_id] = num_blocks
        for i in range(tp_multiplier):
            blocks_data = []
            remote_rank = self.rank * tp_multiplier + i
            for layer_id in range(self.num_layers):
                layer_bases = loc_base[remote_rank][layer_id]
                layer_devids = (loc_dev[remote_rank][layer_id] if loc_dev is not None else None)
                for entry_idx, base_addr in enumerate(layer_bases):
                    rdev = (int(layer_devids[entry_idx]) if layer_devids is not None else int(remote_rank))
                    for block_id in range(num_blocks):
                        block_offset = block_id * dst_block_len
                        blocks_data.append((base_addr + block_offset, dst_block_len, rdev))
            descs = self.nixl_wrapper.get_xfer_descs(blocks_data, "VRAM")
            self.dst_xfer_side_handles[engine_id][i] = self.nixl_wrapper.prep_xfer_dlist(
                self._remote_agents[engine_id][remote_rank], descs
            )
            try:
                if os.getenv("NIXL_CONNECT_WARMUP", "0") == "1":
                    self.nixl_wrapper.make_connection(self._remote_agents[engine_id][remote_rank])
            except Exception as e:
                logger.debug("[ADD] make_connection(%s) lazy: %s", self._remote_agents[engine_id][remote_rank], e)

        logger.info("[ADD] up/equal prepared: src_keys=%s dst_keys=%s dst_num_blocks=%s",
                    list(self.src_xfer_side_handles.keys()),
                    list(self.dst_xfer_side_handles[engine_id].keys()),
                    self.dst_num_blocks[engine_id])
        return agent_names

    _last_done_log_ts = 0.0

    def get_done_tranfers(self) -> List[str]:
        done_req_ids: List[str] = []
        for req_id, handles in list(self._transfers.items()):
            if not isinstance(req_id, str) or req_id == "":
                logger.error("[DONE] illegal key (drop): type=%s repr=%r",
                             type(req_id).__name__, req_id)
                del self._transfers[req_id]
                continue

            running = []
            for h in handles:
                st = self.nixl_wrapper.check_xfer_state(h)
                if st == "DONE":
                    continue
                if st == "PROC":
                    running.append(h)
                else:
                    logger.error("[DONE] transfer failed state=%s (key=%s)", st, req_id)
                    raise RuntimeError(f"[DONE] transfer failed with state {st}")

            if not running:
                done_req_ids.append(req_id)
                del self._transfers[req_id]
            else:
                self._transfers[req_id] = running

        if done_req_ids:
            logger.info("[DONE] report: count=%d keys=%s",
                        len(done_req_ids), done_req_ids[:8])
        else:
            now = time.time()
            if now - getattr(self, "_last_done_log_ts", 0.0) > 1.0:
                logger.debug("[DONE] report: empty")
                self._last_done_log_ts = now

        return done_req_ids
