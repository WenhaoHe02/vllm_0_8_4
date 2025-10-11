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
import hashlib
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

        # ---- engine_id fingerprinting (strict check) ----
        self._engine_fingerprint = {}  # engine_id -> fp
        self._engine_agent_tp = {}     # engine_id -> tp size for check

        # ---- timing ----
        self._timing = _Timing(
            enabled=_env_flag("NIXL_TIMING", True),
            tag=os.getenv("NIXL_TIMING_TAG", f"nixl.{engine_id}.r{rank}")
        )
        self._timing_autolog = _env_flag("NIXL_TIMING_LOG", False)

    # --------- helpers for engine_id check ---------
    def _agents_fingerprint(self, agent_metadata: List[bytes]) -> str:
        h = hashlib.sha1()
        for m in agent_metadata:
            if isinstance(m, bytes):
                h.update(m)
            else:
                h.update(bytes(m))
        return h.hexdigest()

    def _check_engine_id_reuse(self, engine_id: str, agent_metadata: List[bytes], agent_tp: int):
        fp = self._agents_fingerprint(agent_metadata)
        if engine_id in self._engine_fingerprint:
            same_fp = (self._engine_fingerprint[engine_id] == fp)
            same_tp = (self._engine_agent_tp.get(engine_id) == int(agent_tp))
            if not same_fp:
                strict = _env_flag("NIXL_STRICT_ENGINE_ID", True)
                msg = (f"[ENGCHK] engine_id reused with different remote agents: engine_id={engine_id} "
                       f"old_fp={self._engine_fingerprint[engine_id][:12]} new_fp={fp[:12]} "
                       f"old_tp={self._engine_agent_tp.get(engine_id)} new_tp={agent_tp}")
                if strict:
                    logger.error(msg + " -> raising due to NIXL_STRICT_ENGINE_ID=1")
                    raise RuntimeError(msg)
                else:
                    logger.warning(msg + " -> continuing (STRICT=0). Beware of handle overwrite.")
            elif not same_tp:
                logger.warning("[ENGCHK] same engine_id & same fingerprint but TP differs: engine_id=%s old_tp=%s new_tp=%s",
                               engine_id, self._engine_agent_tp.get(engine_id), agent_tp)
        else:
            logger.info("[ENGCHK] first time engine_id=%s fp=%s tp=%s", engine_id, fp[:12], agent_tp)
            self._engine_fingerprint[engine_id] = fp
            self._engine_agent_tp[engine_id] = int(agent_tp)

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

            # 取对端 remote_rank（在该 engine 内部的 TP 槽位），用于句柄选择
            remote_rank = info["remote_rank"]

            # ======= 早期强断言（防止“只预购建在 rank0”导致静默卡死）=======
            # src handle
            if 1 not in self.src_xfer_side_handles or self.src_xfer_side_handles[1] is None:
                msg = (
                    f"[WRITE-DOWN] missing src token handle: "
                    f"engine={self.engine_id} local_rank={self.rank} "
                    f"src_keys={list(self.src_xfer_side_handles.keys())}"
                )
                logger.error(msg)
                raise RuntimeError(msg)

            # dst handle
            if (dst_engine_id not in self.dst_xfer_side_handles or
                remote_rank not in self.dst_xfer_side_handles[dst_engine_id] or
                self.dst_xfer_side_handles[dst_engine_id][remote_rank] is None):
                msg = (
                    f"[WRITE-DOWN] missing dst token handle: "
                    f"dst_engine={dst_engine_id} local_rank={self.rank} remote_rank={remote_rank} "
                    f"dst_keys_top={list(self.dst_xfer_side_handles.keys())} "
                    f"dst_keys_inner={list(self.dst_xfer_side_handles.get(dst_engine_id, {}).keys())}"
                )
                logger.error(msg)
                raise RuntimeError(msg)

            MAX_IOV = int(os.getenv("NIXL_MAX_IOV", "8192"))
            MAX_INFLIGHT = int(os.getenv("NIXL_MAX_INFLIGHT", "4"))
            BACKENDS = ["UCX"] if os.getenv("NIXL_FORCE_UCX", "1") == "1" else None

            src_hdl = self.src_xfer_side_handles[1]
            dst_hdl = self.dst_xfer_side_handles[dst_engine_id][remote_rank]

            # block->token 展开（local / remote 各自展开，保证顺序一致）
            token_ids_remote = self._expand_blocks_to_tokens(remote_block_ids)
            token_ids_local = self._expand_blocks_to_tokens(local_block_ids)
            logger.info("[WRITE-DOWN] token_expand local=%d remote=%d remote_rank=%d", len(token_ids_local), len(token_ids_remote), remote_rank)

            notify_payload = notify_msg if isinstance(notify_msg, str) else str(notify_msg)
            remote_agent = self._remote_agents[dst_engine_id][remote_rank]
            is_leader = bool(info["notify_leader"])

            inflight = []
            total_reqs = 0
            last_req_args = None

            per_entry_src = int(self.num_blocks) * int(self.block_size)
            per_entry_dst = int(self.dst_num_blocks[dst_engine_id])
            logger.debug("[WRITE-DOWN] per_entry src=%d dst=%d num_layers=%d num_entries=%d", per_entry_src, per_entry_dst, self.num_layers, self.num_cache_entries)

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
                        last_req_args = (local_idx, remote_idx)
                        if (hi < N) or (entry < self.num_cache_entries - 1) or (layer < self.num_layers - 1):
                            h = self.nixl_wrapper.make_prepped_xfer(
                                "WRITE",
                                src_hdl, local_idx,
                                dst_hdl, remote_idx,
                                "" if notify_payload is None else ""
                                ,
                                backends=BACKENDS
                            )
                            self.nixl_wrapper.transfer(h)
                            inflight.append(h)
                            total_reqs += 1
                            if len(inflight) >= MAX_INFLIGHT:
                                self._wait_many(inflight)
                                inflight.clear()

            if inflight:
                self._wait_many(inflight)
                inflight.clear()

            # 组内 barrier：只有 leader 等全员
            self._barrier_mark_and_wait(dst_engine_id, notify_payload, info["group_size"], info["peer_idx"], is_leader)

            # 最后一批 piggyback 通知；极端 0-token 情况下仅发通知
            if last_req_args is None:
                if is_leader:
                    logger.info("[NOTIF][DOWN-0tok] dst=%s remote_rank=%s key=%s", dst_engine_id, remote_rank, self._peek([notify_payload]))
                    self.nixl_wrapper.send_notif(remote_agent, notify_payload)
                return

            local_idx, remote_idx = last_req_args
            h_last = self.nixl_wrapper.make_prepped_xfer(
                "WRITE",
                src_hdl, local_idx,
                dst_hdl, remote_idx,
                notify_payload,
                backends=BACKENDS
            )
            self.nixl_wrapper.transfer(h_last)
            self._wait_many([h_last])
            logger.info("[WRITE][DOWN] chunks=%d iov_per_req<=%d inflight<=%d", total_reqs + 1, MAX_IOV, MAX_INFLIGHT)

    def _read_blocks_down(self, local_block_ids, staging_block_ids, remote_block_ids, dst_engine_id):
        down = self._downscale_info[dst_engine_id]
        assert down is not None, "[READ-DOWN] downscale info missing"

        # ======= 早期强断言 =======
        if "read_down_src" not in self.src_xfer_side_handles or self.src_xfer_side_handles["read_down_src"] is None:
            msg = (
                f"[READ-DOWN] missing local READ dst (staging) handle: "
                f"engine={self.engine_id} local_rank={self.rank} keys={list(self.src_xfer_side_handles.keys())}"
            )
            logger.error(msg)
            raise RuntimeError(msg)

        if (dst_engine_id not in self.dst_xfer_side_handles or
            "read_down_dst" not in self.dst_xfer_side_handles[dst_engine_id] or
            self.dst_xfer_side_handles[dst_engine_id]["read_down_dst"] is None):
            msg = (
                f"[READ-DOWN] missing remote READ src handle: "
                f"dst_engine={dst_engine_id} local_rank={self.rank} "
                f"dst_keys_top={list(self.dst_xfer_side_handles.keys())} "
                f"dst_keys_inner={list(self.dst_xfer_side_handles.get(dst_engine_id, {}).keys())}"
            )
            logger.error(msg)
            raise RuntimeError(msg)

        dst_handle = self.src_xfer_side_handles["read_down_src"]      # 本地（prefill）作为 READ 目的地（标准布局）
        src_handle = self.dst_xfer_side_handles[dst_engine_id]["read_down_dst"]  # 远端（decode）作为 READ 来源（按块）

        def _ids_blockwise(num_blocks_total, block_ids):
            ids = []
            for layer in range(self.num_layers):
                for entry in range(self.num_cache_entries):
                    for b in block_ids:
                        ids.append(layer * self.num_cache_entries * num_blocks_total
                                   + entry * num_blocks_total + b)
            return ids

        num_blocks_remote = self.dst_num_blocks_read[dst_engine_id]
        src_desc_ids = _ids_blockwise(num_blocks_remote, remote_block_ids)
        dst_desc_ids = _ids_blockwise(self.num_blocks, staging_block_ids)
        logger.debug("[READ-DOWN] desc_ids len src=%d dst=%d", len(src_desc_ids), len(dst_desc_ids))

        h = self.nixl_wrapper.make_prepped_xfer(
            "READ",
            dst_handle, dst_desc_ids,
            src_handle, src_desc_ids,
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

        ngroups = self._tp_size[self.engine_id] // max(1, self._tp_size[dst_engine_id])
        if ngroups <= 1:
            return

        local_ranges = self._get_ranges(local_block_ids)
        staging_ranges = self._get_ranges(local_block_ids)
        for (l0, l1), (s0, s1) in zip(local_ranges, staging_ranges):
            for kv_cache in self.kv_caches:
                for cache in kv_cache:
                    t_std = cache[s0: s1 + 1].contiguous()
                    t_grp = cache[l0: l1 + 1].contiguous()
                    rearrange_tensors_read_down(t_std, t_grp, ngroups)
                    cache[l0: l1 + 1].copy_(t_grp)

    def _local_token_desc_ids(self, token_ids: List[int]) -> List[int]:
        per_entry = self.num_blocks * self.block_size
        ids = []
        for layer_id in range(self.num_layers):
            for entry_index in range(self.num_cache_entries):
                for tok_id in token_ids:
                    ids.append(layer_id * self.num_cache_entries * per_entry +
                               entry_index * per_entry + tok_id)
        return ids

    def _kv_block_u32sum(self, layer: int, entry_idx: int, block_id: int) -> int:
        t = self.kv_caches[layer][entry_idx][block_id]
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
        """
        组内 barrier（仅 leader 等全员）。新增两个行为：
        - NIXL_DOWN_BARRIER=0 时直接短路（用于快速定位/绕过）。
        - NIXL_DOWN_WAIT_MS 超时（默认 200ms），超时打印缺席 peer 并继续，避免“永等”。
        """
        # 一键关闭（便于快速验证是否与 barrier 有关）
        if os.getenv("NIXL_DOWN_BARRIER", "1") == "0":
            logger.warning("[DOWN-BAR] bypassed by NIXL_DOWN_BARRIER=0 (dst=%s key=%s grp=%d idx=%d leader=%s)",
                           dst_engine_id, self._peek([notify_key]), group_size, peer_idx, is_leader)
            return

        d = self._barrier_dir(dst_engine_id, notify_key, group_size)
        my_flag = os.path.join(d, f"{peer_idx}.ok")
        try:
            with open(my_flag, "w") as f:
                f.write("ok")
        except Exception as e:
            logger.warning("[DOWN-BAR] write flag failed: %s", e)

        if not is_leader:
            return

        # leader 等全员，带超时与缺席打印
        try:
            wait_ms = int(os.getenv("NIXL_DOWN_WAIT_MS", "200"))
            deadline = time.time() + (wait_ms / 1000.0)
            for i in range(group_size):
                flag = os.path.join(d, f"{i}.ok")
                while not os.path.exists(flag):
                    if time.time() > deadline:
                        missing = [j for j in range(group_size) if not os.path.exists(os.path.join(d, f"{j}.ok"))]
                        logger.warning("[DOWN-BAR] timeout (%d ms): expected=%d, missing=%s ; continue",
                                       wait_ms, group_size, missing[:8])
                        # 超时后不再阻塞，允许继续流程，防止卡死
                        break
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

    def _md_cache_dir(self) -> str:
        d = os.getenv("NIXL_MD_CACHE_DIR")
        if not d:
            d = "/dev/shm" if os.path.isdir("/dev/shm") else "/tmp"
        os.makedirs(d, exist_ok=True)
        return d

    def _md_cache_path(self, engine_id: str) -> str:
        safe_engine = self._sanitize_key(engine_id, 32)
        return os.path.join(self._md_cache_dir(), f"nixl_md_{safe_engine}.msgpack")

    def _persist_remote_md_cache(
            self,
            engine_id: str,
            agent_metadata,
            kv_caches_base_addr,
            num_blocks: int,
            kv_caches_dev_ids,
            agent_tp: int,
    ) -> None:
        """把对端元数据写到本机共享缓存，供同机其它 rank 采纳。"""
        try:
            payload = {
                "engine_id": engine_id,
                "agent_tp": int(agent_tp),
                "agent_metadata": agent_metadata,
                "kv_caches_base_addr": kv_caches_base_addr,
                "num_blocks": int(num_blocks),
                "kv_caches_dev_ids": kv_caches_dev_ids,
            }
            b = msgspec.msgpack.encode(payload)
            path = self._md_cache_path(engine_id)
            tmp = f"{path}.tmp.{os.getpid()}"
            with open(tmp, "wb") as f:
                f.write(b)
            os.replace(tmp, path)  # 原子替换
            logger.debug("[MD-CACHE] persisted for engine=%s path=%s size=%dB", engine_id, path, len(b))
        except Exception as e:
            logger.debug("[MD-CACHE] persist failed: %s", e)

    def _adopt_remote_md_from_cache(self, engine_id: str) -> bool:
        """如果还没拿到对端元数据，尝试从共享缓存读取并调用 add_remote_agent。"""
        try:
            path = self._md_cache_path(engine_id)
            if not os.path.exists(path):
                return False
            with open(path, "rb") as f:
                data = msgspec.msgpack.decode(f.read())
            if data.get("engine_id") != engine_id:
                return False
            agent_tp = int(data["agent_tp"])
            # 可能已就绪：再调一次也安全，内部有一致性检查
            self.add_remote_agent(
                engine_id=data["engine_id"],
                agent_metadata=data["agent_metadata"],
                agent_tp=agent_tp,
                kv_caches_base_addr=data["kv_caches_base_addr"],
                num_blocks=int(data["num_blocks"]),
                kv_caches_dev_ids=data.get("kv_caches_dev_ids"),
            )
            logger.info("[MD-CACHE] adopted metadata for engine=%s", engine_id)
            return True
        except Exception as e:
            logger.warning("[MD-CACHE] adopt failed for engine=%s: %s", engine_id, e)
            return False

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
            logger.info("[READ] created_transfers=%s", len(handles))

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

                # --- 长度一致性 ---
                assert len(staging_block_ids) == len(local_block_ids), \
                    f"[WRITE] len mismatch: staging={len(staging_block_ids)} local={len(local_block_ids)}"
                assert len(remote_block_ids) == len(local_block_ids), \
                    f"[WRITE] len mismatch: remote={len(remote_block_ids)} local={len(local_block_ids)}"

                # --- 先确保“本 rank 对 dst_engine”的就绪（包含 adopt + 句柄就绪等待）---
                wait_ms = int(os.getenv("NIXL_READY_WAIT_MS", "3000"))
                self._ensure_engine_ready_for(dst_engine_id, wait_ms)

                # --- 分支：DOWN 优先（因为 add_remote_agent 在 tp_mult==0 会写入 _downscale_info）---
                down = self._downscale_info.get(dst_engine_id)

                def _to_notify_str(x):
                    return x if isinstance(x, str) else str(x)

                if down is not None:
                    # 再做一次快速断言，便于排查
                    info = self._downscale_info[dst_engine_id]
                    rr = info["remote_rank"]
                    if 1 not in self.src_xfer_side_handles or self.src_xfer_side_handles[1] is None:
                        raise RuntimeError(
                            f"[WRITE] DOWN missing src handle (rank={self.rank}) keys={list(self.src_xfer_side_handles.keys())}")
                    if (dst_engine_id not in self.dst_xfer_side_handles or
                            rr not in self.dst_xfer_side_handles[dst_engine_id]):
                        raise RuntimeError(f"[WRITE] DOWN missing dst handle (rank={self.rank} rr={rr}) "
                                           f"dst_keys_top={list(self.dst_xfer_side_handles.keys())} "
                                           f"dst_keys_inner={list(self.dst_xfer_side_handles.get(dst_engine_id, {}).keys())}")

                    self._write_blocks_down(local_block_ids, remote_block_ids, dst_engine_id, notify_msg)

                    if os.getenv("NIXL_DOWN_VERIFY", "1") == "1":
                        try:
                            if remote_block_ids:
                                self._down_verify_peer_segment(dst_engine_id, remote_block_ids[0])
                        except Exception as e:
                            logger.warning("[DOWN-CHK] verify failed: %s", e)

                    logger.info("[WRITE] end ok dst=%s (DOWN)", dst_engine_id)
                    if self._timing_autolog:
                        stats = self.get_timing(reset=True)
                        if stats:
                            logger.info("[TIMING][WRITE-DOWN] %s", stats)
                    return

                # ===== UP / EQ 路径（保持你的原逻辑）=====
                tp_multiplier = self._tp_size[dst_engine_id] // self._tp_size[self.engine_id]
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
                    for i in range(tp_multiplier):
                        trg = self._remote_agents[dst_engine_id][self.rank * tp_multiplier + i]
                        key = _to_notify_str(notify_msg)
                        self.nixl_wrapper.send_notif(trg, key)
                    logger.info("[WRITE] zero-block broadcast sent (tp=%s)", tp_multiplier)
                    if self._timing_autolog:
                        stats = self.get_timing(reset=True)
                        if stats:
                            logger.info("[TIMING][WRITE] %s", stats)
                    return

                if do_rearrange:
                    for l_rng, s_rng in zip(_local_rearranging_ranges, staging_rearranging_ranges):
                        for kv_cache in self.kv_caches:
                            for cache in kv_cache:
                                rearrange_tensors(
                                    cache[l_rng[0]: l_rng[1] + 1],
                                    cache[s_rng[0]: s_rng[1] + 1],
                                    eff_tp, "write"
                                )

                remote_block_descs_ids = self._get_block_descs_ids(dst_engine_id, "all", remote_block_ids)
                local_handle = self.src_xfer_side_handles[tp_multiplier]
                created, handles = 0, []

                notify_payload_str = _to_notify_str(notify_msg)

                for i in targets:
                    staging_block_descs_ids = self._get_block_descs_ids(
                        self.engine_id, "all", staging_block_ids,
                        i=i, tp_multiplier=eff_tp, staging_ranges=staging_rearranging_ranges
                    )
                    if len(staging_block_descs_ids) != len(remote_block_descs_ids):
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
                            raise RuntimeError(f"[WRITE] transfer failed with state {st}")
                    pending = nxt
                    if pending:
                        time.sleep(0.001)

                logger.info("[WRITE] end ok dst=%s (UP/EQ)", dst_engine_id)
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

    def _rebuild_down_dst_handle(self, dst_engine_id: str, remote_rank: int) -> None:
        info = self._downscale_info.get(dst_engine_id)
        assert info is not None, "[REPAIR] downscale info missing"
        group_size = int(info["group_size"])
        peer_idx = int(info["peer_idx"])

        B = int(self.block_size)
        token_len_local = self.block_len // B
        token_len_total = token_len_local * group_size
        full_len = token_len_total * B
        peer_off_tok = peer_idx * token_len_local

        # 选设备 id：优先用 kv_caches_dev_ids，其次用你原来的 engine->gpu 映射
        def _remote_dev_id(r_engine_id, r_idx, layer, entry_idx):
            devs = self.kv_caches_dev_ids.get(r_engine_id)
            if devs is not None:
                return int(devs[r_idx][layer][entry_idx])
            # 回退：保持和 add_remote_agent 相同的映射策略
            if not hasattr(self, "_engine_gpu_map") or r_engine_id not in self._engine_gpu_map:
                # 默认放到 0（或你环境变量里的池），避免 KeyError
                g = int(os.getenv("NIXL_REPAIR_DEFAULT_GPU", "0"))
                if not hasattr(self, "_engine_gpu_map"):
                    self._engine_gpu_map = {}
                self._engine_gpu_map[r_engine_id] = g
            return int(self._engine_gpu_map[r_engine_id])

        # 重新拼装 dst token 粒度的块描述
        dst_blocks = []
        num_blocks = self.dst_num_blocks_read.get(dst_engine_id)  # 注意：READ 用块计数
        if num_blocks is None:
            # 回退：用 tokens 对齐的总单位换回块数（按 B）
            tokens_total = self.dst_num_blocks.get(dst_engine_id)
            assert tokens_total is not None, "[REPAIR] dst units missing"
            num_blocks = tokens_total // B

        for layer in range(self.num_layers):
            layer_bases = self.kv_caches_base_addr[dst_engine_id][remote_rank][layer]
            for entry_idx, rbase in enumerate(layer_bases):  # K / V
                rdev = _remote_dev_id(dst_engine_id, remote_rank, layer, entry_idx)
                for bid in range(num_blocks):
                    base_block = rbase + bid * full_len
                    for t in range(B):
                        dst_blocks.append((base_block + t * token_len_total + peer_off_tok,
                                           token_len_local, rdev))

        dst_desc = self.nixl_wrapper.get_xfer_descs(dst_blocks, "VRAM")
        h = self.nixl_wrapper.prep_xfer_dlist(self._remote_agents[dst_engine_id][remote_rank], dst_desc)

        if dst_engine_id not in self.dst_xfer_side_handles:
            self.dst_xfer_side_handles[dst_engine_id] = {}
        self.dst_xfer_side_handles[dst_engine_id][remote_rank] = h

        try:
            self.nixl_wrapper.make_connection(self._remote_agents[dst_engine_id][remote_rank])
        except Exception:
            pass

        logger.info("[READY][REPAIR] dst handle rebuilt engine=%s rr=%d", dst_engine_id, remote_rank)


    def _ensure_engine_ready_for(self, dst_engine_id: str, wait_ms: int) -> None:
        t0 = time.time()
        adopt_attempts = 0
        last_missing = "unknown"

        def _down_state():
            info = self._downscale_info.get(dst_engine_id)
            if info is None:
                return False, "down_info=None", None
            rr = info.get("remote_rank")
            src_ok = (1 in self.src_xfer_side_handles and self.src_xfer_side_handles[1] is not None)
            dst_ok = (dst_engine_id in self.dst_xfer_side_handles and
                      rr in self.dst_xfer_side_handles[dst_engine_id] and
                      self.dst_xfer_side_handles[dst_engine_id][rr] is not None)
            nb_ok = (dst_engine_id in self.dst_num_blocks)
            ready = (src_ok and dst_ok and nb_ok)
            miss = f"down_ready(src={src_ok}, dst={dst_ok}, nb={nb_ok}, rr={rr})"
            return ready, miss, rr

        def _up_state():
            tp_dst = self._tp_size.get(dst_engine_id)
            tp_src = self._tp_size.get(self.engine_id)
            if tp_dst is None or tp_src is None or tp_src <= 0:
                return False, f"tp_size_missing(dst_has={tp_dst is not None}, src_has={tp_src is not None})"
            tp_mult = tp_dst // tp_src
            eff_tp = max(1, tp_mult)
            src_ok = (tp_mult in self.src_xfer_side_handles and
                      self.src_xfer_side_handles[tp_mult] is not None)
            dst_map = self.dst_xfer_side_handles.get(dst_engine_id) or {}
            dst_ok = all((i in dst_map and dst_map[i] is not None) for i in range(eff_tp))
            nb_ok = (dst_engine_id in self.dst_num_blocks)
            ready = (tp_mult >= 1 and src_ok and dst_ok and nb_ok)
            miss = (f"up_ready(tp_dst={tp_dst}, tp_src={tp_src}, tp_mult={tp_mult}, "
                    f"src={src_ok}, dst={dst_ok}, nb={nb_ok})")
            return ready, miss

        while True:
            down_ready, down_miss, rr = _down_state()
            if down_ready:
                return

            # *** REPAIR: 有 info 但缺 dst 句柄 —— 直接重建这一条句柄 ***
            if self._downscale_info.get(dst_engine_id) is not None and rr is not None:
                dst_ok = (dst_engine_id in self.dst_xfer_side_handles and
                          rr in self.dst_xfer_side_handles[dst_engine_id] and
                          self.dst_xfer_side_handles[dst_engine_id][rr] is not None)
                if not dst_ok:
                    try:
                        self._rebuild_down_dst_handle(dst_engine_id, rr)
                        # 重建成功后立刻复查
                        continue
                    except Exception as e:
                        # 当前 info 可能脏了，清理后交给 adopt
                        logger.warning(
                            "[READY][REPAIR] rebuild dst handle failed engine=%s rr=%s: %s ; purge info and retry adopt",
                            dst_engine_id, rr, e)
                        try:
                            self._downscale_info.pop(dst_engine_id, None)
                        except Exception:
                            pass

            up_ready, up_miss = _up_state()
            if up_ready:
                return

            last_missing = down_miss if self._downscale_info.get(dst_engine_id) is not None else up_miss

            # 未就绪 => 尝试 adopt（不管 _tp_size 是否已存在）
            adopted = False
            try:
                if hasattr(self, "_adopt_remote_md_from_cache"):
                    adopted = self._adopt_remote_md_from_cache(dst_engine_id)
            except Exception as e:
                logger.warning("[READY] adopt failed engine=%s: %s", dst_engine_id, e)
            if adopted:
                adopt_attempts += 1
                logger.info("[READY] adopt success engine=%s (attempt=%d)", dst_engine_id, adopt_attempts)
                continue

            if (time.time() - t0) * 1000.0 > wait_ms:
                raise RuntimeError(
                    f"[WRITE] precondition not met on rank={self.rank} dst={dst_engine_id}: {last_missing} ; "
                    f"_tp_size_keys={list(self._tp_size.keys())} "
                    f"src_keys={list(self.src_xfer_side_handles.keys())} "
                    f"dst_keys_top={list(self.dst_xfer_side_handles.keys())}"
                )
            time.sleep(0.001)

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
            # ---------- 幂等早退：如果这个 engine 已经完全就绪，直接复用 ----------
            if not hasattr(self, "_engine_ready"):
                self._engine_ready = {}
            if self._engine_ready.get(engine_id, False):
                logger.info("[ADD][IDEMPOTENT] reuse engine=%s: src_keys=%s dst_keys=%s down=%s nb=%s",
                            engine_id,
                            list(self.src_xfer_side_handles.keys()),
                            list(self.dst_xfer_side_handles.get(engine_id, {}).keys()),
                            engine_id in self._downscale_info,
                            self.dst_num_blocks.get(engine_id))
                return self._remote_agents[engine_id]

            env_rank = os.getenv("RANK", "?")
            env_local_rank = os.getenv("LOCAL_RANK", "?")
            pid = os.getpid()
            dev = None
            try:
                dev = int(torch.cuda.current_device())
            except Exception:
                pass

            logger.info("[ADD][ENTER] pid=%s env.RANK=%s env.LOCAL_RANK=%s cuda_dev=%s", pid, env_rank, env_local_rank,
                        dev)
            logger.info("[ADD] num_blocks=%d dev_ids=%s", num_blocks, "Y" if kv_caches_dev_ids is not None else "N")
            logger.info("[ADD] engine=%s local_rank=%s local_tp=%s agent_tp=%s is_mla=%s",
                        engine_id, self.rank, self._tp_size[self.engine_id], agent_tp, self._is_mla)

            # strict engine_id check
            self._check_engine_id_reuse(engine_id, agent_metadata, agent_tp)

            # 记录对端 TP
            self._tp_size[engine_id] = int(agent_tp)

            # 持久化元数据（便于同机其它 rank 采用）
            try:
                self._persist_remote_md_cache(
                    engine_id=engine_id,
                    agent_metadata=agent_metadata,
                    kv_caches_base_addr=kv_caches_base_addr,
                    num_blocks=num_blocks,
                    kv_caches_dev_ids=kv_caches_dev_ids,
                    agent_tp=agent_tp,
                )
            except Exception as e:
                logger.debug("[ADD] _persist_remote_md_cache failed: %s", e)

            # 注册远端 agent（只建一次）
            agent_names: List[str] = []
            for meta in agent_metadata:
                agent_names.append(self.nixl_wrapper.add_remote_agent(meta))
            self._remote_agents[engine_id] = agent_names
            logger.info("[ADD] remote_agents registered: dst_engine=%s count=%d names_sample=%s",
                        engine_id, len(agent_names), self._peek(agent_names, 3))

            self.kv_caches_base_addr[engine_id] = kv_caches_base_addr
            self.kv_caches_dev_ids[engine_id] = kv_caches_dev_ids if kv_caches_dev_ids is not None else None
            loc_base = self.kv_caches_base_addr[engine_id]

            # 形状一致性校验（只在首轮）
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

            # -------- 池映射（稳定）--------
            role = "VLLMWORKER" if int(agent_tp) == 1 else "PREFILLWORKER"

            def _parse_pool(env_name: str, default_csv: str) -> list[int]:
                s = os.getenv(env_name, default_csv).strip()
                out = []
                for t in s.split(","):
                    t = t.strip()
                    if t:
                        try:
                            out.append(int(t))
                        except Exception:
                            pass
                return out

            vllm_pool = _parse_pool("NIXL_POOL_VLLMWORKER", "0,1")
            prefill_pool = _parse_pool("NIXL_POOL_PREFILLWORKER", "2,3,4,5")

            if not hasattr(self, "_engine_gpu_map"):
                self._engine_gpu_map = {}  # engine_id -> gpu_id
            if not hasattr(self, "_pool_cursor"):
                self._pool_cursor = {"VLLMWORKER": 0, "PREFILLWORKER": 0}

            def _alloc_gpu_for_engine(eid: str, role: str) -> int:
                if eid in self._engine_gpu_map:
                    return self._engine_gpu_map[eid]
                pool = vllm_pool if role == "VLLMWORKER" else prefill_pool
                if not pool:
                    logger.warning("[ADD][MAP] empty pool for role=%s, fallback gpu=0", role)
                    g = 0
                else:
                    idx = self._pool_cursor[role] % len(pool)
                    g = int(pool[idx])
                    self._pool_cursor[role] += 1
                self._engine_gpu_map[eid] = g
                logger.info("[ADD][MAP] engine=%s role=%s -> gpu_id=%d (pool=%s)", eid, role, g, pool)
                return g

            def _remote_dev_id(r_engine_id: str, r_idx: int, layer: int, entry_idx: int) -> int:
                devs = self.kv_caches_dev_ids.get(r_engine_id)
                if devs is not None:
                    try:
                        return int(devs[r_idx][layer][entry_idx])
                    except Exception:
                        logger.warning("[ADD] invalid kv_caches_dev_ids for engine=%s r=%d L=%d E=%d",
                                       r_engine_id, r_idx, layer, entry_idx)
                return _alloc_gpu_for_engine(r_engine_id, role)

            # --------- DOWN（prefill -> decode，tp_multiplier == 0）---------
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

                # 目的（decode）的 token 粒度单位数
                self.dst_num_blocks[engine_id] = num_blocks * B
                logger.info(
                    "[ADD][DOWN] group_size=%d remote_rank=%d peer_idx=%d token_len_local=%d token_len_total=%d full_len=%d seg_len=%d peer_off_tok=%d",
                    group_size, remote_rank, peer_idx, token_len_local, token_len_total, full_len, seg_len,
                    peer_off_tok)

                # src dlist（仅一次）
                if 1 not in self.src_xfer_side_handles or self.src_xfer_side_handles[1] is None:
                    src_blocks = []
                    local_dev_id = int(torch.cuda.current_device())
                    for layer in range(self.num_layers):
                        for base in self.kv_caches_base_addr[self.engine_id][layer]:  # K、V
                            for bid in range(self.num_blocks):
                                base_block = base + bid * seg_len
                                for t in range(B):
                                    src_blocks.append((base_block + t * token_len_local, token_len_local, local_dev_id))
                    desc = self.nixl_wrapper.get_xfer_descs(src_blocks, "VRAM")
                    self.src_xfer_side_handles[1] = self.nixl_wrapper.prep_xfer_dlist("", desc)

                # dst dlist（仅一次，索引以 remote_rank 锁定）
                if engine_id not in self.dst_xfer_side_handles:
                    self.dst_xfer_side_handles[engine_id] = {}
                if remote_rank not in self.dst_xfer_side_handles[engine_id]:
                    dst_blocks = []
                    for layer in range(self.num_layers):
                        layer_bases = self.kv_caches_base_addr[engine_id][remote_rank][layer]
                        for entry_idx, rbase in enumerate(layer_bases):  # K、V
                            rdev = _remote_dev_id(engine_id, remote_rank, layer, entry_idx)
                            for bid in range(num_blocks):
                                base_block = rbase + bid * full_len
                                for t in range(B):
                                    dst_blocks.append((base_block + t * token_len_total + peer_off_tok,
                                                       token_len_local, rdev))
                    desc = self.nixl_wrapper.get_xfer_descs(dst_blocks, "VRAM")
                    self.dst_xfer_side_handles[engine_id][remote_rank] = self.nixl_wrapper.prep_xfer_dlist(
                        self._remote_agents[engine_id][remote_rank], desc
                    )
                    try:
                        self.nixl_wrapper.make_connection(self._remote_agents[engine_id][remote_rank])
                    except Exception as e:
                        logger.debug("make_connection lazy: %s", e)

                # READ-DOWN 本地目的 dlist（仅一次）
                if "read_down_src" not in self.src_xfer_side_handles or self.src_xfer_side_handles[
                    "read_down_src"] is None:
                    blocks_local = []
                    local_dev_id = int(torch.cuda.current_device())
                    for layer in range(self.num_layers):
                        for base in self.kv_caches_base_addr[self.engine_id][layer]:
                            for bid in range(self.num_blocks):
                                block_offset = bid * self.block_len
                                blocks_local.append((base + block_offset, self.block_len, local_dev_id))
                    descs_local = self.nixl_wrapper.get_xfer_descs(blocks_local, "VRAM")
                    self.src_xfer_side_handles["read_down_src"] = self.nixl_wrapper.prep_xfer_dlist("", descs_local)

                # READ-DOWN 远端来源 dlist（仅一次）
                if "read_down_dst" not in self.dst_xfer_side_handles[engine_id]:
                    blocks_remote = []
                    for layer in range(self.num_layers):
                        layer_bases = self.kv_caches_base_addr[engine_id][remote_rank][layer]  # [K_base, V_base]
                        for entry_idx, rbase in enumerate(layer_bases):
                            rdev = _remote_dev_id(engine_id, remote_rank, layer, entry_idx)
                            for bid in range(num_blocks):
                                block_offset = bid * self.block_len
                                blocks_remote.append((rbase + block_offset, self.block_len, rdev))
                    descs_remote = self.nixl_wrapper.get_xfer_descs(blocks_remote, "VRAM")
                    self.dst_xfer_side_handles[engine_id]["read_down_dst"] = self.nixl_wrapper.prep_xfer_dlist(
                        self._remote_agents[engine_id][remote_rank], descs_remote
                    )
                    try:
                        self.nixl_wrapper.make_connection(self._remote_agents[engine_id][remote_rank])
                    except Exception as e:
                        logger.debug("[ADD][READ-DOWN] make_connection lazy: %s", e)

                if not hasattr(self, "dst_num_blocks_read"):
                    self.dst_num_blocks_read = {}
                self.dst_num_blocks_read[engine_id] = num_blocks

                logger.info(
                    "[ADD][DOWN][READY] engine=%s local_rank=%d remote_rank=%d src_keys=%s dst_keys=%s dst_units(token)=%s read_down_keys=%s",
                    engine_id, self.rank, remote_rank,
                    list(self.src_xfer_side_handles.keys()),
                    list(self.dst_xfer_side_handles[engine_id].keys()),
                    self.dst_num_blocks[engine_id],
                    list(self.dst_xfer_side_handles[engine_id].keys()))

                self._engine_ready[engine_id] = True
                return agent_names

            # --------- UP / EQ（tp_multiplier > 0）---------
            assert tp_multiplier > 0, f"[ADD] invalid tp_multiplier={tp_multiplier}"
            dst_block_len = self.block_len if self._is_mla else (self.block_len // tp_multiplier)
            logger.info("[ADD] up/equal path: dst_block_len=%s", dst_block_len)

            if tp_multiplier not in self.src_xfer_side_handles or self.src_xfer_side_handles[tp_multiplier] is None:
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
            if engine_id not in self.dst_xfer_side_handles:
                self.dst_xfer_side_handles[engine_id] = {}
            for i in range(tp_multiplier):
                if i in self.dst_xfer_side_handles[engine_id] and self.dst_xfer_side_handles[engine_id][i] is not None:
                    continue  # 已建则复用
                blocks_data = []
                remote_idx = self.rank * tp_multiplier + i
                for layer_id in range(self.num_layers):
                    layer_bases = loc_base[remote_idx][layer_id]
                    for entry_idx, base_addr in enumerate(layer_bases):
                        for block_id in range(num_blocks):
                            block_offset = block_id * dst_block_len
                            blocks_data.append((base_addr + block_offset, dst_block_len, int(remote_idx)))
                descs = self.nixl_wrapper.get_xfer_descs(blocks_data, "VRAM")
                self.dst_xfer_side_handles[engine_id][i] = self.nixl_wrapper.prep_xfer_dlist(
                    self._remote_agents[engine_id][remote_idx], descs
                )
                try:
                    self.nixl_wrapper.make_connection(self._remote_agents[engine_id][remote_idx])
                except Exception as e:
                    logger.debug("make_connection lazy: %s", e)

            logger.info("[ADD][UP][READY] engine=%s local_rank=%d src_keys=%s dst_keys=%s dst_num_blocks=%s",
                        engine_id, self.rank,
                        list(self.src_xfer_side_handles.keys()),
                        list(self.dst_xfer_side_handles[engine_id].keys()),
                        self.dst_num_blocks[engine_id])

            self._engine_ready[engine_id] = True
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
