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

import torch
from typing import List, Tuple, Optional
from vllm.config import VllmConfig
from vllm.logger import init_logger
import os
import msgspec
import time
import uuid
from collections import defaultdict
from .kv_rearrange import rearrange_tensors

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
        dict=True):
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

        self.kv_caches_dev_ids = {}

        self._transfers = defaultdict(list)

        self._tp_size[engine_id] = vllm_config.parallel_config.tensor_parallel_size
        self._is_mla = "deepseek" in vllm_config.model_config.architectures[0].lower()

        self.block_size = None
        self.head_dim = None

        # used only when prefill TP > decode TP
        self._downscale_info = {}

    def _expand_blocks_to_tokens(self, block_ids: list[int]) -> list[int]:
        # 展开：block_id -> block_id*block_size + [0..block_size-1]
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

    def _write_blocks_down(self, local_block_ids, remote_block_ids, dst_engine_id, notify_msg):
        down = self._downscale_info[dst_engine_id]
        assert down is not None, "[WRITE-DOWN] downscale info missing"

        # 1) 展开到 token 粒度
        token_ids = self._expand_blocks_to_tokens(remote_block_ids)
        # 本地同样是按 token 切好的 dlist（你现有 add_remote_agent(DOWN) 已经用 token_len_local 生成了）
        staging_token_ids = self._expand_blocks_to_tokens(local_block_ids)

        # 2) 取 token 粒度的 desc id
        remote_desc_ids = self._get_block_descs_ids(dst_engine_id, "all", token_ids)  # 注意：这里 token_ids
        local_handle = self.src_xfer_side_handles[1]  # DOWN 固定用 key=1（token 粒度）
        remote_handle = self.dst_xfer_side_handles[dst_engine_id][0]

        # 3) 发送（DOWN：DMA 不带 notify；notify 独立在 barrier 后，仅 leader 发）
        h = self.nixl_wrapper.make_prepped_xfer(
            "WRITE",
            local_handle, self._get_block_descs_ids(self.engine_id, "all", staging_token_ids, i=0, tp_multiplier=1),
            remote_handle, remote_desc_ids,
            ""  # 不带 payload
        )
        self.nixl_wrapper.transfer(h)

        # 等 DONE
        while True:
            st = self.nixl_wrapper.check_xfer_state(h)
            if st == "DONE":
                break
            if st != "PROC":
                raise RuntimeError(f"[WRITE-DOWN] transfer failed: {st}")
            time.sleep(0.001)

        # barrier + 最终通知（仅 leader）
        payload = notify_msg if isinstance(notify_msg, str) else str(notify_msg)
        self._barrier_mark_and_wait(dst_engine_id, payload, down["group_size"], down["peer_idx"], down["notify_leader"])
        if down["notify_leader"]:
            trg = self._remote_agents[dst_engine_id][down["remote_rank"]]
            self.nixl_wrapper.send_notif(trg, payload)

    def _read_blocks_down(self, local_block_ids, remote_block_ids, dst_engine_id):
        down = self._downscale_info[dst_engine_id]
        assert down is not None, "[READ-DOWN] downscale info missing"

        # 1) 一样展开到 token 粒度（保持 1:1 长度）
        token_ids = self._expand_blocks_to_tokens(remote_block_ids)
        staging_token_ids = self._expand_blocks_to_tokens(local_block_ids)  # 目标写到本地这些 token 切片

        # 2) desc
        local_handle = self.src_xfer_side_handles[1]
        remote_handle = self.dst_xfer_side_handles[dst_engine_id][0]
        remote_desc_ids = self._get_block_descs_ids(dst_engine_id, "all", token_ids)
        staging_desc_ids = self._get_block_descs_ids(self.engine_id, "all", staging_token_ids, i=0, tp_multiplier=1)

        # 3) 读（纯 DMA）
        h = self.nixl_wrapper.make_prepped_xfer(
            "READ",
            local_handle, staging_desc_ids,
            remote_handle, remote_desc_ids,
            ""
        )
        self.nixl_wrapper.transfer(h)
        while True:
            st = self.nixl_wrapper.check_xfer_state(h)
            if st == "DONE":
                break
            if st != "PROC":
                raise RuntimeError(f"[READ-DOWN] transfer failed: {st}")
            time.sleep(0.001)

    def _local_token_desc_ids(self, token_ids: list[int]) -> list[int]:
        """为 DOWN 场景构造本地 src 句柄的 token 粒度 desc 索引。
           顺序需与 add_remote_agent(DOWN) 里构造 src_desc 的循环一致：
             for layer in L:
               for entry in [K,V]:
                 for bid in range(num_blocks):
                   for t in range(B):
                     append(...)
        """
        per_entry = self.num_blocks * self.block_size  # 每个 entry 的“单位”数：block×token
        ids = []
        for layer_id in range(self.num_layers):
            for entry_index in range(self.num_cache_entries):
                for tok_id in token_ids:
                    ids.append(layer_id * self.num_cache_entries * per_entry +
                               entry_index * per_entry + tok_id)
        return ids

    def _kv_block_u32sum(self, layer: int, entry_idx: int, block_id: int) -> int:
        """对某层、某 entry(K=0/V=1)、某 block 的本地段做 32 位有符号整形求和（GPU->CPU）。
           只用于调试校验，不改变数据。
        """
        # 非 MLA：self.kv_caches[layer] 是 (key_cache, value_cache)
        t = self.kv_caches[layer][entry_idx][block_id]  # shape: [block_size, num_heads_local, head_dim]
        # 视图成 int32 做和，降低体积&对齐；结果取 python int
        return int(t.view(torch.int32).sum().item())

    def _down_verify_peer_segment(self, dst_engine_id: str,
                                  remote_block_id: int,
                                  scratch_block_id: Optional[int] = None,
                                  max_layers: int = 2) -> None:
        """将远端 remote_block_id 的“本 peer 段”读回到本地一个 scratch block，
           然后分别对 K/V 做 32 位和校验，与本地原 block 对比。
           注意：read_blocks(DOWN) 会自动按 token 粒度展开。
        """
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
            logger.info("[DOWN-CHK] engine=%s layer=%d block=%d -> scratch=%d K:%d==%d %s V:%d==%d %s",
                        dst_engine_id, layer, remote_block_id, scratch_block_id,
                        src_k, dst_k, "OK" if src_k == dst_k else "MISMATCH",
                        src_v, dst_v, "OK" if src_v == dst_v else "MISMATCH")
        logger.info("[DOWN-CHK] summary: K=%s V=%s (layers checked=%d)",
                    "OK" if k_ok else "MISMATCH",
                    "OK" if v_ok else "MISMATCH", L)

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
        logger.debug("Is deepseek: %s", self._is_mla)
        try:
            # 尝试打印 sample shapes safely
            if self._is_mla:
                logger.debug("kv_cache sample shape (mla): %s", getattr(kv_caches[0], "shape", None))
            else:
                # kv_caches[layer] == (key_cache, value_cache)
                k0 = kv_caches[0][0] if isinstance(kv_caches[0], (list, tuple)) else kv_caches[0]
                v0 = kv_caches[0][1] if isinstance(kv_caches[0], (list, tuple)) else None
                logger.debug("kv_cache sample key shape: %s value shape: %s", getattr(k0, "shape", None),
                             getattr(v0, "shape", None))
        except Exception as e:
            logger.warning("[KVREG] preview shapes failed: %s", e)
        logger.debug("--------------------------------")

        if self._is_mla:
            # MLA path (kept simple)
            num_blocks, block_size, head_dim = kv_caches[0].shape
            self.block_len = head_dim * block_size * kv_caches[0].element_size()
            logger.debug("Per layer kv cache size (mla): %s", kv_caches[0].shape)
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
            # 非 MLA，期望 kv_caches[layer] == (key_cache, value_cache)
            # 保护性读取 shape & elem_size
            try:
                key0 = kv_caches[0][0]
                val0 = kv_caches[0][1]
            except Exception as e:
                logger.exception("[KVREG] kv_caches structure unexpected: %s", e)
                raise

            num_blocks, block_size, num_heads, head_dim = key0.shape
            elem_size = key0.element_size()  # 字节
            dtype = getattr(key0, "dtype", None)

            # block_len in bytes = block_size * num_heads * head_dim * elem_size
            self.block_len = int(block_size * num_heads * head_dim * elem_size)
            self.block_size = int(block_size)
            self.head_dim = int(head_dim)
            logger.info("[KVREG] key dtype=%s elem_size=%d bytes", dtype, elem_size)

            self.num_layers = len(kv_caches)
            self.num_blocks = int(num_blocks)
            self.num_heads = int(num_heads)
            self.kv_caches = kv_caches
            self.num_cache_entries = 2

            kv_caches_base_addr = []
            caches_data = []
            for key_cache, value_cache in kv_caches:
                base_addr = int(key_cache.data_ptr())
                region_len = int(self.num_cache_entries * self.num_blocks * self.block_len)
                caches_data.append((base_addr, region_len, self.rank, ""))
                kv_caches_base_addr.append([int(key_cache.data_ptr()), int(value_cache.data_ptr())])

            self.kv_caches_base_addr[self.engine_id] = kv_caches_base_addr
            self.kv_caches_dev_ids.setdefault(self.engine_id, None)

            # debug 打印一些关键数字，便于校验
            logger.info(
                "[KVREG] engine=%s layers=%d blocks=%d entries=%d block_len=%dB elem=%d heads=%d head_dim=%d block_size=%d",
                self.engine_id, self.num_layers, self.num_blocks, self.num_cache_entries,
                self.block_len, elem_size, self.num_heads, self.head_dim, self.block_size)
            logger.debug("[KVREG] sample base addrs (first layer): %s",
                         kv_caches_base_addr[0] if kv_caches_base_addr else None)

            descs = self.nixl_wrapper.get_reg_descs(caches_data, "VRAM")
            logger.debug("Registering descs: %s", caches_data[:3])
            self.nixl_wrapper.register_memory(descs)
            self._registered_descs.append(descs)

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

    def _get_ranges(self, block_ids):
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
                            if block_id > staging_ranges[staging_range_idx][1] or block_id < staging_ranges[staging_range_idx][0]:
                                staging_range_idx += 1
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

    def _peek(xs, k=3):
        return xs[:k] + (["..."] if len(xs) > k else [])
    def read_blocks(self, local_block_ids, staging_block_ids, remote_block_ids, dst_engine_id):
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
            return self._read_blocks_down(local_block_ids, remote_block_ids, dst_engine_id)
            eff_tp, targets = 1, [0]
            logger.info("[READ] downscale active: remote_rank=%s group_size=%s -> eff_tp=1 targets=%s",
                        downscale_info.get("remote_rank"), downscale_info.get("group_size"), targets)
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
            self.nixl_wrapper.transfer(handle)
            handles.append(handle)
        logger.info("[READ] created_transfers=%s create_ms=%.3f",
                    len(handles), (time.perf_counter() - t0) * 1000.0)

        t1 = time.perf_counter()
        pending = list(handles)
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
        logger.info("[READ] transfer_ms=%.3f", (time.perf_counter() - t1) * 1000.0)

        t2 = time.perf_counter()
        if not self._is_mla:
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
        logger.info("[READ] rearrange_ms=%.3f total_ms=%.3f",
                    (time.perf_counter() - t2) * 1000.0,
                    (time.perf_counter() - start_time) * 1000.0)

    def write_blocks(self, local_block_ids, staging_block_ids, remote_block_ids, dst_engine_id, notify_msg):
        try:
            logger.info("[WRITE] begin dst=%s local=%d staging=%d remote=%d notify_type=%s",
                        dst_engine_id, len(local_block_ids), len(staging_block_ids),
                        len(remote_block_ids), type(notify_msg).__name__)

            remote_block_ids = remote_block_ids[:len(local_block_ids)]
            assert len(staging_block_ids) == len(local_block_ids), \
                f"[WRITE] len mismatch: staging={len(staging_block_ids)} local={len(local_block_ids)}"

            down = self._downscale_info.get(dst_engine_id)
            tp_multiplier = self._tp_size[dst_engine_id] // self._tp_size[self.engine_id]

            def _to_notify_str(x):
                return x if isinstance(x, str) else str(x)

            notify_payload_str = "" if (down is not None) else _to_notify_str(notify_msg)

            # ========= DOWN 分支 =========
            if down is not None:
                self._write_blocks_down(local_block_ids, remote_block_ids, dst_engine_id, notify_msg)
                rr = down["remote_rank"]
                group_size = down.get("group_size") or (
                            self._tp_size[self.engine_id] // max(1, self._tp_size[dst_engine_id]))
                peer_idx = down.get("peer_idx", self.rank % group_size)
                leader = down.get("notify_leader", (peer_idx == 0))
                logger.info("[WRITE] path=DOWN remote_rank=%s group_size=%s peer_idx=%s leader=%s",
                            rr, group_size, peer_idx, leader)

                # block id → token id 展开
                B = int(self.block_size)

                def _expand_to_token_ids(block_ids: List[int], B: int) -> List[int]:
                    out = []
                    for bid in block_ids:
                        base = bid * B
                        out.extend([base + t for t in range(B)])
                    return out

                staging_tok_ids = _expand_to_token_ids(staging_block_ids, B)
                remote_tok_ids = _expand_to_token_ids(remote_block_ids, B)

                # desc 索引
                local_xfer_side_handle = self.src_xfer_side_handles[1]  # 我们在 DOWN add_remote_agent 里建的 token 粒度句柄
                staging_block_descs_ids = self._local_token_desc_ids(staging_tok_ids)
                remote_block_descs_ids = self._get_block_descs_ids(dst_engine_id, "all", remote_tok_ids)

                assert len(staging_block_descs_ids) == len(remote_block_descs_ids), \
                    f"[WRITE-DOWN] desc mismatch: staging={len(staging_block_descs_ids)} remote={len(remote_block_descs_ids)}"
                if dst_engine_id not in self.dst_xfer_side_handles:
                    raise RuntimeError(f"[WRITE-DOWN] missing dst_xfer_side_handles for {dst_engine_id}")
                remote_xfer_side_handle = self.dst_xfer_side_handles[dst_engine_id][0]

                # 发 DMA（不带 notify）
                h = self.nixl_wrapper.make_prepped_xfer(
                    "WRITE",
                    local_xfer_side_handle, staging_block_descs_ids,
                    remote_xfer_side_handle, remote_block_descs_ids,
                    ""
                )
                self.nixl_wrapper.transfer(h)

                # 等 DONE
                t1 = time.perf_counter()
                while True:
                    st = self.nixl_wrapper.check_xfer_state(h)
                    if st == "DONE":
                        break
                    if st != "PROC":
                        logger.error("[WRITE-DOWN] transfer failed state=%s", st)
                        raise RuntimeError(f"[WRITE-DOWN] transfer failed with state {st}")
                    time.sleep(0.001)
                logger.info("[WRITE-DOWN] local_xfer_wait_ms=%.3f", (time.perf_counter() - t1) * 1000)

                # 组内 barrier，再由 leader 发送最终 notify
                try:
                    payload_key = _to_notify_str(notify_msg)
                    self._barrier_mark_and_wait(dst_engine_id, payload_key, group_size, peer_idx, leader)
                    logger.info("[DOWN-BAR] group ready: peers=%s key=%s", group_size, payload_key)
                except Exception as e:
                    logger.warning("[DOWN-BAR] barrier failed (continue anyway): %s", e)

                if leader:
                    trg = self._remote_agents[dst_engine_id][rr]
                    payload = _to_notify_str(notify_msg)
                    try:
                        self.nixl_wrapper.send_notif(trg, payload)
                        logger.info("[WRITE] final notify sent -> remote_rank=%s len=%d", rr, len(payload))
                    except Exception as e:
                        logger.exception("[WRITE] final notify error: %s", e)
                        raise

                logger.info("[WRITE] end ok dst=%s", dst_engine_id)
                if os.getenv("NIXL_DOWN_VERIFY", "1") == "1":
                    try:
                        if remote_block_ids:
                            self._down_verify_peer_segment(dst_engine_id, remote_block_ids[0])
                    except Exception as e:
                        logger.warning("[DOWN-CHK] verify failed: %s", e)
                return

            # ========= UP/EQ 分支（原逻辑） =========
            eff_tp = max(1, tp_multiplier)
            targets = list(range(eff_tp))

            do_rearrange = False
            staging_rearranging_ranges = None
            if not self._is_mla:
                local_ranges = self._get_ranges(local_block_ids)
                staging_ranges = self._get_ranges(staging_block_ids)
                _local_rearranging_ranges, staging_rearranging_ranges = self._get_same_length_ranges(
                    local_ranges, staging_ranges)
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
                            rearrange_tensors(
                                cache[l_rng[0]: l_rng[1] + 1],
                                cache[s_rng[0]: s_rng[1] + 1],
                                eff_tp, "write"
                            )
                logger.info("[WRITE] rearrange_ms=%.3f", (time.perf_counter() - t0) * 1000)

            remote_block_descs_ids = self._get_block_descs_ids(dst_engine_id, "all", remote_block_ids)
            local_handle = self.src_xfer_side_handles[eff_tp]
            created, handles = 0, []
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

                h = self.nixl_wrapper.make_prepped_xfer(
                    "WRITE",
                    local_handle, staging_block_descs_ids,
                    remote_handle, remote_block_descs_ids,
                    notify_payload_str
                )
                if notify_payload_str:
                    self._transfers.setdefault(notify_payload_str, []).append(h)
                self.nixl_wrapper.transfer(h)
                handles.append(h)
                created += 1

            t1 = time.perf_counter()
            pending = list(handles)
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
            logger.info("[WRITE] local_xfer_wait_ms=%.3f", (time.perf_counter() - t1) * 1000)

            logger.info("[WRITE] end ok dst=%s", dst_engine_id)

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
        logger.info("[ADD] num_blocks=%d dev_ids=%s", num_blocks, "Y" if kv_caches_dev_ids is not None else "N")
        logger.info("[ADD] engine=%s local_rank=%s local_tp=%s agent_tp=%s is_mla=%s",
                    engine_id, self.rank, self._tp_size[self.engine_id], agent_tp, self._is_mla)

        self._tp_size[engine_id] = int(agent_tp)

        # 注册远端 agents
        agent_names: List[str] = []
        for meta in agent_metadata:
            agent_names.append(self.nixl_wrapper.add_remote_agent(meta))
        self._remote_agents[engine_id] = agent_names

        # 保存远端 KV 基址/设备表
        self.kv_caches_base_addr[engine_id] = kv_caches_base_addr
        self.kv_caches_dev_ids[engine_id] = kv_caches_dev_ids if kv_caches_dev_ids is not None else None
        loc_base = self.kv_caches_base_addr[engine_id]
        loc_dev = self.kv_caches_dev_ids[engine_id]

        # 一致性校验
        assert len(agent_metadata) == agent_tp
        assert len(loc_base) == agent_tp
        for r in range(agent_tp):
            assert len(loc_base[r]) == self.num_layers
            for L in range(self.num_layers):
                assert len(loc_base[r][L]) == self.num_cache_entries

        tp_multiplier = self._tp_size[engine_id] // self._tp_size[self.engine_id]
        logger.info("[ADD] tp_multiplier=%s (dst_tp/src_tp = %s/%s)",
                    tp_multiplier, self._tp_size[engine_id], self._tp_size[self.engine_id])

        # ====================
        # DOWN：prefill TP > decode TP，且非 MLA
        # ====================
        if tp_multiplier == 0 and not self._is_mla:
            group_size = self._tp_size[self.engine_id] // max(1, self._tp_size[engine_id])
            remote_rank = self.rank // group_size
            peer_idx = self.rank % group_size
            slot = peer_idx  # 不做换位

            # 尺度（字节）
            # block_len = B * H_local * C * elem
            B = int(self.block_size)
            token_len_local = self.block_len // B  # = H_local * C * bytes
            token_len_total = token_len_local * group_size  # = H_total * C * bytes
            seg_len = token_len_local * B  # = B * H_local * C * bytes
            full_len = token_len_total * B  # = B * H_total * C * bytes
            peer_off_tok = slot * token_len_local

            # 记录元信息
            self._downscale_info[engine_id] = {
                "group_size": group_size,
                "remote_rank": remote_rank,
                "peer_idx": peer_idx,
                "notify_leader": (peer_idx == 0),
                "perm": None,
                "token_granularity": True,
            }

            # ***** 关键：远端 desc 的“单位数”按 token 粒度 *****
            self.dst_num_blocks[engine_id] = num_blocks * B

            logger.info(
                "[ADD][DOWN] group_size=%d remote_rank=%d peer_idx=%d token_len_local=%d token_len_total=%d full_len=%d seg_len=%d",
                group_size, remote_rank, peer_idx, token_len_local, token_len_total, full_len, seg_len)

            # --- 构造本地（prefill）SRC 描述符：按 token 粒度 ---
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

            # --- 构造远端（decode）DST 描述符：按 token 粒度 ---
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
                            # 注意：同一 token 内不同 peer 仅在 token 内偏移不同
                            dst_blocks.append((base_block + t * token_len_total + peer_off_tok,
                                               token_len_local, int(rdev)))
            dst_desc = self.nixl_wrapper.get_xfer_descs(dst_blocks, "VRAM")
            self.dst_xfer_side_handles[engine_id][0] = self.nixl_wrapper.prep_xfer_dlist(
                self._remote_agents[engine_id][remote_rank], dst_desc
            )

            # 懒连接
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

        # ====================
        # UP/EQ：保持原逻辑
        # ====================
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
                self.nixl_wrapper.make_connection(self._remote_agents[engine_id][remote_rank])
            except Exception as e:
                logger.debug("[ADD] make_connection(%s) lazy: %s", self._remote_agents[engine_id][remote_rank], e)

        logger.info("[ADD] up/equal prepared: src_keys=%s dst_keys=%s dst_num_blocks=%s",
                    list(self.src_xfer_side_handles.keys()),
                    list(self.dst_xfer_side_handles[engine_id].keys()),
                    self.dst_num_blocks[engine_id])
        return agent_names

    _last_done_log_ts = 0.0  # 放在类的 __init__ 里也可

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

        # 只有有完成项才打 info；空结果限频/降级
        if done_req_ids:
            logger.info("[DONE] report: count=%d keys=%s",
                        len(done_req_ids), done_req_ids[:8])
        else:
            # 可选：限频 1 秒一条；或者直接 logger.debug
            now = time.time()
            if now - getattr(self, "_last_done_log_ts", 0.0) > 1.0:
                logger.debug("[DONE] report: empty")
                self._last_done_log_ts = now

        return done_req_ids

