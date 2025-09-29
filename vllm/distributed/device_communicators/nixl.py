# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import uuid
from collections import defaultdict
from typing import List, Tuple, Optional

import msgspec
import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger

from .kv_rearrange import rearrange_tensors, rearrange_tensors_read_down
from contextlib import contextmanager
from functools import lru_cache

logger = init_logger(__name__)

# Lazy import nixl_wrapper to avoid loading nixl_bindings if nixl is not used
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
    kv_caches_base_addr: List[List[List[int]]]  # base address for each rank for each layer for keys and values
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
        self._remote_index_map = {}

        self.kv_caches_dev_ids = {}

        self._transfers = defaultdict(list)

        self._tp_size[engine_id] = vllm_config.parallel_config.tensor_parallel_size
        self._is_mla = "deepseek" in vllm_config.model_config.architectures[0].lower()

        self.block_size = None
        self.head_dim = None

        self._downscale_info = {}

        # ---- timing ----
        self._timing = _Timing(
            enabled=_env_flag("NIXL_TIMING", True),
            tag=os.getenv("NIXL_TIMING_TAG", f"nixl.{engine_id}.r{rank}")
        )
        self._timing_autolog = _env_flag("NIXL_TIMING_LOG", False)

    def _wait_many(self, handles):
        with self._timing.span("wait_many"):
            # 轻量轮询 + 退避
            spins, SPIN_MAX = 0, 2000
            sleep_us, SLEEP_MAX = 200, 2000
            pending = list(handles)
            while pending:
                nxt = []
                for h in pending:
                    st = self.nixl_wrapper.check_xfer_state(h)
                    if st == "DONE":
                        continue
                    if st != "PROC":
                        raise RuntimeError(f"[DOWN] transfer failed: {st}")
                    nxt.append(h)
                if not nxt:
                    return
                pending = nxt
                if spins < SPIN_MAX:
                    spins += 1
                else:
                    time.sleep(sleep_us / 1e6)
                    sleep_us = min(sleep_us * 2, SLEEP_MAX)

    def _chunk_iter(self, total_len: int, chunk: int):
        off = 0
        while off < total_len:
            end = min(off + chunk, total_len)
            yield off, end
            off = end

    @lru_cache(maxsize=1024)
    def _expand_seq(self, start_block: int, n_blocks: int) -> Tuple[int, ...]:
        B = int(self.block_size)
        return tuple([t for b in range(start_block, start_block + n_blocks) for t in range(b * B, b * B + B)])

    def _expand_blocks_to_tokens(self, block_ids: List[int]) -> List[int]:
        with self._timing.span("expand_blocks_to_tokens"):
            if not block_ids:
                return []
            # 连续段合并
            rngs = self._get_ranges(block_ids)
            out = []
            for a, b in rngs:
                out.extend(self._expand_seq(a, b - a + 1))
            return out

    def _write_blocks_down(self, local_block_ids, remote_block_ids, dst_engine_id, notify_msg):
        with self._timing.span("write_down"):
            info = self._downscale_info[dst_engine_id]
            assert info is not None, "[WRITE-DOWN] downscale info missing"

            # 先拿到 remote_rank，再校验并取句柄（修复：句柄按 remote_rank 分槽）
            remote_rank = info["remote_rank"]

            MAX_IOV = int(os.getenv("NIXL_MAX_IOV", "8192"))
            MAX_INFLIGHT = int(os.getenv("NIXL_MAX_INFLIGHT", "4"))
            BACKENDS = ["UCX"] if os.getenv("NIXL_FORCE_UCX", "1") == "1" else None

            assert 1 in self.src_xfer_side_handles, "[WRITE-DOWN] missing src token handle"
            assert dst_engine_id in self.dst_xfer_side_handles and remote_rank in self.dst_xfer_side_handles[dst_engine_id], \
                "[WRITE-DOWN] missing dst token handle"

            src_hdl = self.src_xfer_side_handles[1]
            dst_hdl = self.dst_xfer_side_handles[dst_engine_id][remote_rank]

            # 把“这次要搬的 block 列表”展开为 token 索引（单位：token）
            # 注意：我们对 local/remote 分别展开，然后用同样的层×entry 顺序拼索引，保证一一对应
            token_ids_remote = self._expand_blocks_to_tokens(remote_block_ids)
            token_ids_local = self._expand_blocks_to_tokens(local_block_ids)

            # 构造“本次传输”的索引序列，但不一次性传给 API，而是下面按 MAX_IOV 切片
            # 顺序 = 按 layer → entry(K/V) → token
            # （与 add_remote_agent 里预构 dlist 的顺序完全一致）
            # 为了避免一次性分配 40+ 万长度的大列表带来的 Python 压力，
            # 这里逐层生成并切片发送。
            notify_payload = notify_msg if isinstance(notify_msg, (bytes, bytearray)) else str(notify_msg).encode()
            remote_agent = self._remote_agents[dst_engine_id][remote_rank]
            is_leader = bool(info["notify_leader"])

            inflight = []
            total_reqs = 0
            last_req_args = None  # 留给最终“带通知”的那一小批

            # 逐层/逐 entry 生成索引并切片
            per_entry_src = int(self.num_blocks) * int(self.block_size)  # 本地 6553 * 64 = 419,392
            per_entry_dst = int(self.dst_num_blocks[dst_engine_id])
            # 将当前选择的 token_ids（相对某层/某entry）映射到全局 dlist 的索引：
            # idx = layer * (num_entries * per_entry) + entry * per_entry + tok_id

            segments = [(lo, hi, token_ids_local[lo:hi], token_ids_remote[lo:hi])
                        for lo, hi in self._chunk_iter(len(token_ids_local), MAX_IOV)]
            N = len(token_ids_local)
            for layer in range(self.num_layers):
                base_layer_src = layer * (self.num_cache_entries * per_entry_src)
                base_layer_dst = layer * (self.num_cache_entries * per_entry_dst)
                for entry in range(self.num_cache_entries):
                    base_entry_src = base_layer_src + entry * per_entry_src
                    base_entry_dst = base_layer_dst + entry * per_entry_dst
                    for (lo, hi, seg_local, seg_remote) in segments:
                        local_idx = [base_entry_src + t for t in seg_local]
                        remote_idx = [base_entry_dst + t for t in seg_remote]
                        # 最后一批留到 barrier 之后再发，且 piggyback 通知
                        last_req_args = (local_idx, remote_idx)
                        # 其余先发（不带通知）
                        # 通过滑动窗口控制并发
                        if (hi < N) or (entry < self.num_cache_entries - 1) or (layer < self.num_layers - 1):
                            h = self.nixl_wrapper.make_prepped_xfer(
                                "WRITE",
                                src_hdl, local_idx,
                                dst_hdl, remote_idx,
                                b"",  # 不带 payload
                                backends=BACKENDS
                            )
                            self.nixl_wrapper.transfer(h)
                            inflight.append(h)
                            total_reqs += 1
                            if len(inflight) >= MAX_INFLIGHT:
                                self._wait_many(inflight)
                                inflight.clear()

            # 等非最后一批全部完成
            t_wait0 = time.perf_counter_ns()
            if inflight:
                self._wait_many(inflight)
                inflight.clear()
            self._timing.add("write_down.wait_bulk_ns", time.perf_counter_ns() - t_wait0)

            # 组内 barrier：只有 leader 等全员到齐
            t_bar0 = time.perf_counter_ns()
            self._barrier_mark_and_wait(dst_engine_id, notify_payload, info["group_size"], info["peer_idx"], is_leader)
            self._timing.add("write_down.barrier_ns", time.perf_counter_ns() - t_bar0)

            # 发送“最后一小批 + 通知”
            if last_req_args is None:
                # 极端：这次实际上没有要搬的 token，leader 仍需发通知防止 decode 端等待
                if is_leader:
                    self.nixl_wrapper.send_notif(remote_agent, notify_payload)
                return

            local_idx, remote_idx = last_req_args
            h_last = self.nixl_wrapper.make_prepped_xfer(
                "WRITE",
                src_hdl, local_idx,
                dst_hdl, remote_idx,
                notify_payload,  # 只在最后一批带通知
                backends=BACKENDS
            )
            t_send_last0 = time.perf_counter_ns()
            self.nixl_wrapper.transfer(h_last)
            self._wait_many([h_last])
            self._timing.add("write_down.last_send_ns", time.perf_counter_ns() - t_send_last0)
            logger.info("[WRITE][DOWN] chunks=%d iov_per_req<=%d inflight<=%d", total_reqs + 1, MAX_IOV, MAX_INFLIGHT)

    def _read_blocks_down(self, local_block_ids, staging_block_ids, remote_block_ids, dst_engine_id):
        down = self._downscale_info[dst_engine_id]
        assert down is not None, "[READ-DOWN] downscale info missing"

        # 句柄（块粒度）
        dst_handle = self.src_xfer_side_handles["read_down_src"]  # 本地（prefill）作为 READ 目的地
        src_handle = self.dst_xfer_side_handles[dst_engine_id]["read_down_dst"]  # 远端（decode）作为 READ 来源

        # 用“块”为单位生成 dlist 索引（顺序：按 layer→entry→block）
        def _ids_blockwise(num_blocks_total, block_ids):
            ids = []
            for layer in range(self.num_layers):
                for entry in range(self.num_cache_entries):  # K、V
                    for b in block_ids:
                        ids.append(layer * self.num_cache_entries * num_blocks_total
                                   + entry * num_blocks_total + b)
            return ids

        num_blocks_remote = self.dst_num_blocks_read[dst_engine_id]
        src_desc_ids = _ids_blockwise(num_blocks_remote, remote_block_ids)  # 远端块索引
        dst_desc_ids = _ids_blockwise(self.num_blocks, staging_block_ids)  # 本地 staging 块索引（标准布局）

        # 传输（READ）
        h = self.nixl_wrapper.make_prepped_xfer(
            "READ",
            dst_handle, dst_desc_ids,
            src_handle, src_desc_ids,
            b""
        )
        self.nixl_wrapper.transfer(h)
        while True:
            st = self.nixl_wrapper.check_xfer_state(h)
            if st == "DONE":
                break
            if st != "PROC":
                raise RuntimeError(f"[READ-DOWN] transfer failed: {st}")
            time.sleep(0.001)

        # ===== 传完在本机做重排：standard -> grouped =====
        # ngroups = tp_prefill // tp_decode
        ngroups = self._tp_size[self.engine_id] // max(1, self._tp_size[dst_engine_id])
        if ngroups <= 1:
            return

        # 对本次涉及到的块区间做 GPU 重排（staging -> final）
        # 注意：worker 调用 read_blocks(local, staging, remote, ...) 时，
        #       local_block_ids = 目标（grouped），staging_block_ids = 临时（标准）。
        # 这里我们按 ranges 成批跑 kernel。
        local_ranges = self._get_ranges(local_block_ids)
        staging_ranges = self._get_ranges(local_block_ids)  # 与上面 1:1 对拷

        for (l0, l1), (s0, s1) in zip(local_ranges, staging_ranges):
            # 每个层、每个 entry（K/V）分别重排
            for kv_cache in self.kv_caches:
                # kv_cache: (K,V) or [KV] depending on layout
                for cache in kv_cache:
                    # 切 [N=BID 区间, B, H, C]
                    t_std = cache[s0: s1 + 1].contiguous()  # staging（标准）
                    t_grp = cache[l0: l1 + 1].contiguous()  # 目标（组段）
                    # 目标必须写回原位（可以先 alloc 一块临时，再 copy 回去；为了简单直接 in-place 覆盖）
                    rearrange_tensors_read_down(t_std, t_grp, ngroups)
                    # 把重排结果写回原 cache 位置
                    cache[l0: l1 + 1].copy_(t_grp)

    def _local_token_desc_ids(self, token_ids: List[int]) -> List[int]:
        per_entry = self.num_blocks * self.block_size  # 每个 entry 的“单位”数：block×token
        ids = []
        for layer_id in range(self.num_layers):
            for entry_index in range(self.num_cache_entries):
                for tok_id in token_ids:
                    ids.append(layer_id * self.num_cache_entries * per_entry +
                               entry_index * per_entry + tok_id)
        return ids

    def _kv_block_u32sum(self, layer: int, entry_idx: int, block_id: int) -> int:

        # 微观读性能影响极小，无需分段计时
        t = self.kv_caches[layer][entry_idx][block_id]  # shape: [block_size, num_heads_local, head_dim]
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

        # 触发一次 token 粒度 READ（内部会展开）
        self.read_blocks(
            local_block_ids=[scratch_block_id],
            staging_block_ids=[scratch_block_id],
            remote_block_ids=[remote_block_id],
            dst_engine_id=dst_engine_id,
        )

        # sum32 校验（抽前 max_layers 层）
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
                    time.sleep(0.001)
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

    def register_kv_caches(self, kv_caches: List[torch.Tensor]):
        logger.debug("--------------------------------")
        logger.debug("Registering kv caches for engine %s", self.engine_id)
        logger.debug(f"Is deepseek: {self._is_mla}")
        logger.debug(f"kv_cache shape: {kv_caches[0].shape}")
        logger.debug("--------------------------------")

        if self._is_mla:
            num_blocks, block_size, head_dim = kv_caches[0].shape
            self.block_len = head_dim * block_size * kv_caches[0].element_size()
            logger.debug("Per layer kv cache size: %s", kv_caches[0].shape)
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
            logger.debug("Registering descs: %s", caches_data)
            self.nixl_wrapper.register_memory(descs)
            self._registered_descs.append(descs)
        else:
            _, num_blocks, block_size, num_heads, head_dim = kv_caches[0].shape
            self.block_len = block_size * num_heads * head_dim * kv_caches[0].element_size()
            self.block_size = block_size
            self.head_dim = head_dim
            logger.debug("Per layer kv cache size: %s", kv_caches[0].shape)
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
            logger.debug("Registering descs: %s", caches_data)
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
            # 远端：按对端的 dst_num_blocks（单位：block；若 DOWN 预处理成 token 数）
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

    def read_blocks(self, local_block_ids, staging_block_ids, remote_block_ids, dst_engine_id):
        with self._timing.span("read_blocks"):
            logger.info("[READ] local=%s staging=%s remote=%s dst_engine=%s",
                        len(local_block_ids), len(staging_block_ids), len(remote_block_ids), dst_engine_id)
            assert len(local_block_ids) == len(staging_block_ids) == len(remote_block_ids), \
                f"[READ] len mismatch: local={len(local_block_ids)} staging={len(staging_block_ids)} remote={len(remote_block_ids)}"
            if len(local_block_ids) == 0:
                logger.info("[READ] no-op (0 blocks)")
                return

            start_time = time.perf_counter()
            if self._is_mla:
                staging_rearranging_ranges = None
                staging_block_ids = local_block_ids
            else:
                local_ranges = self._get_ranges(local_block_ids)
                staging_ranges = self._get_ranges(staging_block_ids)
                local_rearranging_ranges, staging_rearranging_ranges = self._get_same_length_ranges(local_ranges,
                                                                                                    staging_ranges)
                logger.debug("[READ] local_ranges=%s staging_ranges=%s -> rearr_local=%s rearr_staging=%s",
                             local_ranges, staging_ranges, local_rearranging_ranges, staging_rearranging_ranges)

            downscale_info = self._downscale_info.get(dst_engine_id)
            tp_multiplier = self._tp_size[dst_engine_id] // self._tp_size[self.engine_id]
            if downscale_info is not None:
                self._read_blocks_down(local_block_ids, staging_block_ids, remote_block_ids, dst_engine_id)
                # 自动或手动打印计时
                if self._timing_autolog:
                    stats = self.get_timing(reset=True)
                    if stats:
                        logger.info("[TIMING][READ-DOWN] %s", stats)
                return
            else:
                eff_tp = max(1, tp_multiplier)
                targets = list(range(eff_tp))
                logger.info("[READ] tp_multiplier=%s eff_tp=%s targets=%s", tp_multiplier, eff_tp, targets)

            remote_block_descs_ids = self._get_block_descs_ids(dst_engine_id, "all", remote_block_ids)
            local_xfer_side_handle = self.src_xfer_side_handles[eff_tp]
            logger.debug("[READ] remote_desc_ids_len=%s local_handle_key=%s", len(remote_block_descs_ids), eff_tp)
            if dst_engine_id not in self.dst_xfer_side_handles:
                raise RuntimeError(f"[READ] dst_xfer_side_handles missing for engine {dst_engine_id}")

            handles = []
            t0 = time.perf_counter()
            for i in targets:
                staging_block_descs_ids = self._get_block_descs_ids(
                    self.engine_id, "all", staging_block_ids, i=i, tp_multiplier=eff_tp,
                    staging_ranges=staging_rearranging_ranges
                )
                assert len(staging_block_descs_ids) == len(remote_block_descs_ids), \
                    f"[READ] desc len mismatch: staging={len(staging_block_descs_ids)} remote={len(remote_block_descs_ids)}"
                remote_xfer_side_handle = self.dst_xfer_side_handles[dst_engine_id][i]
                logger.debug("[READ] i=%s staging_desc_ids_len=%s staging_head=%s remote_head=%s",
                             i, len(staging_block_descs_ids), self._peek(staging_block_descs_ids), self._peek(remote_block_descs_ids))
                handle = self.nixl_wrapper.make_prepped_xfer(
                    "READ",
                    local_xfer_side_handle, staging_block_descs_ids,
                    remote_xfer_side_handle, remote_block_descs_ids,
                    ""
                )
                t_make0 = time.perf_counter_ns()
                self.nixl_wrapper.transfer(handle)
                self._timing.add("read_blocks.submit_ns", time.perf_counter_ns() - t_make0)
                handles.append(handle)
            logger.info("[READ] created_transfers=%s create_ms=%.3f",
                        len(handles), (time.perf_counter() - t0) * 1000.0)

            t1 = time.perf_counter()
            pending = list(handles)
            wait_ns0 = time.perf_counter_ns()
            while pending:
                nxt = []
                for h in pending:
                    status = self.nixl_wrapper.check_xfer_state(h)
                    if status == "DONE":
                        continue
                    elif status == "PROC":
                        nxt.append(h)
                    else:
                        logger.error("[READ] transfer failed: state=%s", status)
                        raise RuntimeError(f"[READ] transfer failed with state {status}")
                pending = nxt
                if pending:
                    time.sleep(0.001)
            self._timing.add("read_blocks.wait_ns", time.perf_counter_ns() - wait_ns0)
            logger.info("[READ] transfer_ms=%.3f", (time.perf_counter() - t1) * 1000.0)

            t2 = time.perf_counter()
            if not self._is_mla:
                re_ns0 = time.perf_counter_ns()
                for local_range, staging_range in zip(local_rearranging_ranges, staging_rearranging_ranges):
                    logger.debug("[READ] rearrange cache_shape=%s local=%s staging=%s eff_tp=%s",
                                 getattr(self.kv_caches[0], "shape", None), local_range, staging_range, eff_tp)
                    for kv_cache in self.kv_caches:
                        for cache in kv_cache:
                            rearrange_tensors(
                                cache[local_range[0]:local_range[1] + 1],
                                cache[staging_range[0]:staging_range[1] + 1],
                                eff_tp, "read"
                            )
                self._timing.add("read_blocks.rearrange_ns", time.perf_counter_ns() - re_ns0)
            logger.info("[READ] rearrange_ms=%.3f total_ms=%.3f",
                        (time.perf_counter() - t2) * 1000.0,
                        (time.perf_counter() - start_time) * 1000.0)

            # 自动或手动打印计时
            if self._timing_autolog:
                stats = self.get_timing(reset=True)
                if stats:
                    logger.info("[TIMING][READ] %s", stats)

    def write_blocks(self, local_block_ids, staging_block_ids, remote_block_ids, dst_engine_id, notify_msg):
        with self._timing.span("write_blocks"):
            try:
                logger.info("[WRITE] begin dst=%s local=%d staging=%d remote=%d notify_type=%s",
                            dst_engine_id, len(local_block_ids), len(staging_block_ids),
                            len(remote_block_ids), type(notify_msg).__name__)

                # 强一致长度检查
                assert len(staging_block_ids) == len(local_block_ids), \
                    f"[WRITE] len mismatch: staging={len(staging_block_ids)} local={len(local_block_ids)}"
                assert len(remote_block_ids) == len(local_block_ids), \
                    f"[WRITE] len mismatch: remote={len(remote_block_ids)} local={len(local_block_ids)}"

                down = self._downscale_info.get(dst_engine_id)
                tp_multiplier = self._tp_size[dst_engine_id] // self._tp_size[self.engine_id]

                def _to_notify_str(x):
                    return x if isinstance(x, str) else str(x)

                if down is not None:
                    self._write_blocks_down(local_block_ids, remote_block_ids, dst_engine_id, notify_msg)
                    if os.getenv("NIXL_DOWN_VERIFY", "1") == "1":
                        try:
                            if remote_block_ids:
                                self._down_verify_peer_segment(dst_engine_id, remote_block_ids[0])
                        except Exception as e:
                            logger.warning("[DOWN-CHK] verify failed: %s", e)
                    logger.info("[WRITE] end ok dst=%s (DOWN)", dst_engine_id)
                    # 自动或手动打印计时
                    if self._timing_autolog:
                        stats = self.get_timing(reset=True)
                        if stats:
                            logger.info("[TIMING][WRITE-DOWN] %s", stats)
                    return

                # ========= UP/EQ 分支 =========
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
                    # 自动或手动打印计时
                    if self._timing_autolog:
                        stats = self.get_timing(reset=True)
                        if stats:
                            logger.info("[TIMING][WRITE] %s", stats)
                    return

                if do_rearrange:
                    t0 = time.perf_counter()
                    re_ns0 = time.perf_counter_ns()
                    for l_rng, s_rng in zip(_local_rearranging_ranges, staging_rearranging_ranges):
                        for kv_cache in self.kv_caches:
                            for cache in kv_cache:
                                rearrange_tensors(
                                    cache[l_rng[0]: l_rng[1] + 1],
                                    cache[s_rng[0]: s_rng[1] + 1],
                                    eff_tp, "write"
                                )
                    logger.info("[WRITE] rearrange_ms=%.3f", (time.perf_counter() - t0) * 1000)
                    self._timing.add("write_blocks.rearrange_ns", time.perf_counter_ns() - re_ns0)

                remote_block_descs_ids = self._get_block_descs_ids(dst_engine_id, "all", remote_block_ids)
                local_handle = self.src_xfer_side_handles[eff_tp]
                created, handles = 0, []

                notify_payload_str = _to_notify_str(notify_msg)

                for i in targets:
                    staging_block_descs_ids = self._get_block_descs_ids(
                        self.engine_id, "all", staging_block_ids,
                        i=i, tp_multiplier=eff_tp, staging_ranges=staging_rearranging_ranges
                    )
                    if len(staging_block_descs_ids) != len(remote_block_descs_ids):
                        logger.error("[WRITE] desc mismatch staging=%d remote=%d (i=%d)",
                                     len(staging_block_descs_ids), len(remote_block_descs_ids), i)
                        raise RuntimeError("desc length mismatch")
                    remote_handle = self.dst_xfer_side_handles[dst_engine_id][i]

                    t_submit0 = time.perf_counter_ns()
                    h = self.nixl_wrapper.make_prepped_xfer(
                        "WRITE",
                        local_handle, staging_block_descs_ids,
                        remote_handle, remote_block_descs_ids,
                        notify_payload_str  # UP/EQ：允许 sideband 一起带
                    )
                    if notify_payload_str:
                        self._transfers.setdefault(notify_payload_str, []).append(h)
                    self.nixl_wrapper.transfer(h)
                    self._timing.add("write_blocks.submit_ns", time.perf_counter_ns() - t_submit0)
                    handles.append(h)
                    created += 1

                t1 = time.perf_counter()
                pending = list(handles)
                wait_ns0 = time.perf_counter_ns()
                while pending:
                    nxt = []
                    for h in pending:
                        st = self.nixl_wrapper.check_xfer_state(h)
                        if st == "DONE":
                            continue
                        if st == "PROC":
                            nxt.append(h)
                        else:
                            logger.error("[WRITE] transfer failed state=%s", st)
                            raise RuntimeError(f"[WRITE] transfer failed with state {st}")
                    pending = nxt
                    if pending:
                        time.sleep(0.001)
                self._timing.add("write_blocks.wait_ns", time.perf_counter_ns() - wait_ns0)
                logger.info("[WRITE] local_xfer_wait_ms=%.3f", (time.perf_counter() - t1) * 1000)
                logger.info("[WRITE] end ok dst=%s (UP/EQ)", dst_engine_id)

                # 自动或手动打印计时
                if self._timing_autolog:
                    stats = self.get_timing(reset=True)
                    if stats:
                        logger.info("[TIMING][WRITE] %s", stats)

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
        with self._timing.span("add_remote_agent"):
            logger.info("[ADD] num_blocks=%d dev_ids=%s", num_blocks, "Y" if kv_caches_dev_ids is not None else "N")
            logger.info("[ADD] engine=%s local_rank=%s local_tp=%s agent_tp=%s is_mla=%s",
                engine_id, self.rank, self._tp_size[self.engine_id], agent_tp, self._is_mla)

            self._tp_size[engine_id] = int(agent_tp)

            # 注册远端 agents
            agent_names: List[str] = []
            for meta in agent_metadata:
                agent_names.append(self.nixl_wrapper.add_remote_agent(meta))
            self._remote_agents[engine_id] = agent_names

            self.kv_caches_base_addr[engine_id] = kv_caches_base_addr
            self.kv_caches_dev_ids[engine_id] = kv_caches_dev_ids if kv_caches_dev_ids is not None else None
            loc_base = self.kv_caches_base_addr[engine_id]
            loc_dev = self.kv_caches_dev_ids[engine_id]

            if len(agent_metadata) != agent_tp:
                raise RuntimeError(f"[ADD] agent_metadata len={len(agent_metadata)} != agent_tp={agent_tp}")
            if len(loc_base) != agent_tp:
                raise RuntimeError(f"[ADD] kv_caches_base_addr outer len={len(loc_base)} != agent_tp={agent_tp}")

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
                token_len_local = self.block_len // B  # = H_local * C * bytes
                token_len_total = token_len_local * group_size  # = H_total * C * bytes
                seg_len = token_len_local * B  # = B * H_local * C * bytes
                full_len = token_len_total * B  # = B * H_total * C * bytes
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

                logger.info(
                    "[ADD][DOWN] group_size=%d remote_rank=%d peer_idx=%d token_len_local=%d token_len_total=%d full_len=%d seg_len=%d",
                    group_size, remote_rank, peer_idx, token_len_local, token_len_total, full_len, seg_len)

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
                t_prep0 = time.perf_counter_ns()
                # 修复：目的 dlist 句柄按 remote_rank 存储，而不是固定 key=0
                self.dst_xfer_side_handles[engine_id][remote_rank] = self.nixl_wrapper.prep_xfer_dlist(
                    self._remote_agents[engine_id][remote_rank], dst_desc
                )
                self._timing.add("add_remote_agent.down_prep_dst_ns", time.perf_counter_ns() - t_prep0)

                # 懒连接
                try:
                    self.nixl_wrapper.make_connection(self._remote_agents[engine_id][remote_rank])
                except Exception as e:
                    logger.debug("make_connection(%s) lazy: %s", self._remote_agents[engine_id][remote_rank], e)

                if "read_down_src" not in self.src_xfer_side_handles:
                    blocks_local = []
                    local_dev_id = int(torch.cuda.current_device())
                    # 本地（prefill 侧）标准布局的每个 block 的 (base, len, dev)
                    for layer in range(self.num_layers):
                        # K、V 两个 entry
                        for base in self.kv_caches_base_addr[self.engine_id][layer]:
                            for bid in range(self.num_blocks):
                                block_offset = bid * self.block_len
                                blocks_local.append((base + block_offset, self.block_len, local_dev_id))
                    descs_local = self.nixl_wrapper.get_xfer_descs(blocks_local, "VRAM")
                    self.src_xfer_side_handles["read_down_src"] = self.nixl_wrapper.prep_xfer_dlist("", descs_local)

                # 远端（decode 侧）按块
                if engine_id not in self.dst_xfer_side_handles:
                    self.dst_xfer_side_handles[engine_id] = {}
                # 选择与本 rank 对应的 remote_rank（你原来 down 代码里已经算过）
                remote_rank = self._downscale_info[engine_id]["remote_rank"]

                blocks_remote = []
                remote_dev_table = self.kv_caches_dev_ids.get(engine_id)
                for layer in range(self.num_layers):
                    layer_bases = self.kv_caches_base_addr[engine_id][remote_rank][layer]  # [K_base, V_base]
                    layer_dev_ids = None
                    if remote_dev_table is not None:
                        layer_dev_ids = remote_dev_table[remote_rank][layer]
                    for entry_idx, rbase in enumerate(layer_bases):
                        rdev = int(layer_dev_ids[entry_idx]) if layer_dev_ids is not None else int(remote_rank)
                        for bid in range(num_blocks):
                            block_offset = bid * self.block_len
                            blocks_remote.append((rbase + block_offset, self.block_len, rdev))

                descs_remote = self.nixl_wrapper.get_xfer_descs(blocks_remote, "VRAM")
                # 专用 key 存起来（与 token 路径区分开）
                self.dst_xfer_side_handles[engine_id]["read_down_dst"] = self.nixl_wrapper.prep_xfer_dlist(
                    self._remote_agents[engine_id][remote_rank], descs_remote
                )

                # 单独记录“read_down 的 num_blocks = 按块数”
                if not hasattr(self, "dst_num_blocks_read"):
                    self.dst_num_blocks_read = {}
                self.dst_num_blocks_read[engine_id] = num_blocks

                # 提前拨号一次（懒连接容错）
                try:
                    self.nixl_wrapper.make_connection(self._remote_agents[engine_id][remote_rank])
                except Exception as e:
                    logger.debug("[ADD][READ-DOWN] make_connection lazy: %s", e)

                logger.info("[ADD] downscale prepared: src_keys=%s dst_keys=%s dst_units(token)=%s",
                            list(self.src_xfer_side_handles.keys()),
                            list(self.dst_xfer_side_handles[engine_id].keys()),
                            self.dst_num_blocks[engine_id])
                return agent_names

            assert tp_multiplier > 0, f"[ADD] invalid tp_multiplier={tp_multiplier}"
            dst_block_len = self.block_len if self._is_mla else (self.block_len // tp_multiplier)
            logger.info("[ADD] up/equal path: dst_block_len=%s", dst_block_len)

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
                t_prep0 = time.perf_counter_ns()
                self.src_xfer_side_handles[tp_multiplier] = self.nixl_wrapper.prep_xfer_dlist("", descs)
                self._timing.add("add_remote_agent.up_prep_src_ns", time.perf_counter_ns() - t_prep0)

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
                t_prep1 = time.perf_counter_ns()
                self.dst_xfer_side_handles[engine_id][i] = self.nixl_wrapper.prep_xfer_dlist(
                    self._remote_agents[engine_id][remote_rank], descs
                )
                self._timing.add("add_remote_agent.up_prep_dst_ns", time.perf_counter_ns() - t_prep1)
                try:
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
        with self._timing.span("get_done_transfers"):
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

    # -------- public timing API --------
    def get_timing(self, reset: bool = False):
        """返回当前累计的计时统计（单位 ns / ms / 平均值），可选清零。"""
        stats = self._timing.snapshot(reset=reset)
        if stats:
            logger.debug("[TIMING] %s", stats)
        return stats


# ===== helpers & timing class =====
def _env_flag(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip() not in ("0", "false", "False", "")


class _Timing:
    """轻量计时聚合：with timing.span('key'): ... / timing.add('key_ns', ns)"""
    __slots__ = ("_enabled", "_ns", "_n", "_tag")

    def __init__(self, enabled: bool, tag: str = "nixl"):
        self._enabled = bool(enabled)
        self._ns = defaultdict(int)   # key -> total ns
        self._n = defaultdict(int)    # key -> calls
        self._tag = tag

    @contextmanager
    def span(self, key: str):
        if not self._enabled:
            yield
            return
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            dt = time.perf_counter_ns() - t0
            self._ns[key] += dt
            self._n[key] += 1

    def add(self, key: str, ns: int):
        if not self._enabled:
            return
        self._ns[key] += int(ns)
        self._n[key] += 1

    def snapshot(self, reset: bool = False):
        if not self._enabled:
            return {}
        out = {}
        for k, tot in self._ns.items():
            n = max(1, self._n.get(k, 1))
            out[f"{self._tag}.{k}.ns"] = int(tot)
            out[f"{self._tag}.{k}.ms"] = round(tot / 1e6, 3)
            out[f"{self._tag}.{k}.avg_ms"] = round((tot / n) / 1e6, 3)
            out[f"{self._tag}.{k}.calls"] = int(self._n.get(k, 0))
        if reset:
            self._ns.clear()
            self._n.clear()
        return out
