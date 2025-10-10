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

    def write_blocks(
            self,
            local_block_ids,  # List[int]
            staging_block_ids,  # List[int]
            remote_block_ids,  # List[int]
            dst_engine_id: str,  # 目标 decode engine_id
            notify_msg,  # bytes or str
    ):
        """
        统一写 KV 的入口。
        仅对 DOWN 路径增加就绪闸门，避免在目标 engine 尚未完成 add_remote_agent()
        以及 dlist 准备前就发起写，造成 precondition 报错。
        """

        import time

        def _down_ready(engine_id: str) -> bool:
            # 是否已有 tp_size、downscale_info、dlist 与 num_blocks_read
            if engine_id not in self._tp_size:
                return False
            if not hasattr(self, "_downscale_info") or engine_id not in self._downscale_info:
                return False
            di = self._downscale_info[engine_id]
            rr = di.get("remote_rank", None)
            if rr is None:
                return False
            # 目的 dlist（token 粒度的 dst_xfer_side_handles[engine_id][remote_rank]）与 read_down_dst
            if engine_id not in self.dst_xfer_side_handles:
                return False
            if rr not in self.dst_xfer_side_handles[engine_id]:
                return False
            if "read_down_dst" not in self.dst_xfer_side_handles[engine_id]:
                return False
            # 目的 num_blocks_read
            if not hasattr(self, "dst_num_blocks_read") or engine_id not in self.dst_num_blocks_read:
                return False
            return True

        def _ensure_down_ready(engine_id: str, timeout_ms: int = 3000) -> None:
            if _down_ready(engine_id):
                return
            # 若本地缓存了对端 MD，尝试一次幂等补建
            md_cache = getattr(self, "_remote_md_cache", {})
            if engine_id in md_cache and engine_id not in getattr(self, "_remote_agents", {}):
                md = md_cache[engine_id]
                try:
                    self.add_remote_agent(
                        engine_id=engine_id,
                        agent_metadata=md["agent_metadata"],
                        agent_tp=int(md["agent_tp"]),
                        kv_caches_base_addr=md["kv_caches_base_addr"],
                        num_blocks=int(md["num_blocks"]),
                        kv_caches_dev_ids=md.get("kv_caches_dev_ids"),
                    )
                except Exception as e:
                    logger.debug("[WRITE][DOWN] add_remote_agent-from-cache failed: %s", e)

            # 短轮询等待另一侧建好
            deadline = time.time() + timeout_ms / 1000.0
            while time.time() < deadline:
                if _down_ready(engine_id):
                    return
                time.sleep(0.005)

            # 仍未就绪——保持原样式报错，方便对齐日志检查
            raise RuntimeError(
                "[WRITE] precondition not met on rank=%d dst=%s: down_ready(src=%s, dst=%s, nb=%s, rr=%s) ; "
                "_tp_size_keys=%s src_keys=%s dst_keys_top=%s" % (
                    self.rank,
                    engine_id,
                    True,
                    False,
                    False,
                    (getattr(self._downscale_info.get(engine_id, {}), "get", lambda *_: None)("remote_rank") if hasattr(
                        self, "_downscale_info") else None),
                    list(self._tp_size.keys()),
                    list(self.src_xfer_side_handles.keys()),
                    (list(self.dst_xfer_side_handles.get(engine_id, {}).keys())
                     if engine_id in self.dst_xfer_side_handles else [])
                )
            )

        try:
            notify_type = type(notify_msg).__name__
            logger.info(
                "[WRITE] begin dst=%s local=%d staging=%d remote=%d notify_type=%s",
                dst_engine_id, int(bool(local_block_ids)), int(bool(staging_block_ids)),
                int(bool(remote_block_ids)), notify_type
            )

            tp_src = self._tp_size.get(self.engine_id, None)
            tp_dst = self._tp_size.get(dst_engine_id, None)
            tp_mult = (None if (tp_src is None or tp_dst is None) else (tp_dst // tp_src))
            down_path = (tp_dst is not None) and (tp_src is not None) and (tp_dst < tp_src) and (not self._is_mla)

            logger.info(
                "[WRITE] path choose: down=%s tp_src=%s tp_dst=%s tp_mult=%s rank=%d local=%d staging=%d remote=%d notify_repr=%r",
                bool(down_path), tp_src, tp_dst, (tp_mult if tp_mult is not None else 0),
                self.rank, int(bool(local_block_ids)), int(bool(staging_block_ids)), int(bool(remote_block_ids)),
                (notify_msg if isinstance(notify_msg, str) else b"<bytes>")
            )

            if down_path or (tp_dst is None):
                # —— 关键：确保 Down 侧映射/句柄已建好（remote_rank 由 add_remote_agent 写入内部状态，不从这里传参）——
                _ensure_down_ready(dst_engine_id, timeout_ms=3000)

                # 正确的 5 个位置参数调用
                return self._write_blocks_down(
                    local_block_ids,
                    remote_block_ids,
                    dst_engine_id,
                    notify_msg,
                )

            # UP/EQ：同样用 5 个位置参数
            return self._write_blocks_up_equal(
                local_block_ids,
                staging_block_ids,
                remote_block_ids,
                dst_engine_id,
                notify_msg,
            )

        except Exception as e:
            logger.error(
                "[WRITE] exception dst=%s down=%s tp_src=%s tp_dst=%s tp_mult=%s rank=%d local=%d staging=%d remote=%d notify_repr=%r",
                dst_engine_id,
                bool(down_path) if 'down_path' in locals() else False,
                tp_src if 'tp_src' in locals() else None,
                tp_dst if 'tp_dst' in locals() else None,
                (tp_mult if ('tp_mult' in locals() and tp_mult is not None) else 0),
                self.rank,
                int(bool(local_block_ids)), int(bool(staging_block_ids)), int(bool(remote_block_ids)),
                (notify_msg if isinstance(notify_msg, str) else b"<bytes>")
            )
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
            # ---------- 严格检查 engine_id 重用 ----------
            self._check_engine_id_reuse(engine_id, agent_metadata, agent_tp)

            # 记录对端 tp 大小
            self._tp_size[engine_id] = int(agent_tp)

            # 持久化远端 KV 结构（地址/设备ID）
            self._persist_remote_md_cache(
                engine_id=engine_id,
                agent_metadata=agent_metadata,
                kv_caches_base_addr=kv_caches_base_addr,
                num_blocks=num_blocks,
                kv_caches_dev_ids=kv_caches_dev_ids,
                agent_tp=agent_tp,
            )

            # ---------- 注册远端 agents（每个 engine 的 remote_rank: 0..agent_tp-1） ----------
            agent_names: List[str] = []
            for meta in agent_metadata:
                agent_names.append(self.nixl_wrapper.add_remote_agent(meta))
            self._remote_agents[engine_id] = agent_names
            logger.info("[ADD] remote_agents registered: dst_engine=%s count=%d names_sample=%s",
                        engine_id, len(agent_names), self._peek(agent_names, 3))

            # 本地缓存这些远端 KV 基址/设备
            self.kv_caches_base_addr[engine_id] = kv_caches_base_addr
            self.kv_caches_dev_ids[engine_id] = kv_caches_dev_ids if kv_caches_dev_ids is not None else None
            loc_base = self.kv_caches_base_addr[engine_id]

            # ---------- 形状自检 ----------
            if len(agent_metadata) != agent_tp:
                raise RuntimeError(f"[ADD] agent_metadata len={len(agent_metadata)} != agent_tp={agent_tp}")
            if len(loc_base) != agent_tp:
                raise RuntimeError(f"[ADD] kv_caches_base_addr outer len={len(loc_base)} != agent_tp={agent_tp}")
            for r in range(agent_tp):
                assert len(loc_base[r]) == self.num_layers
                for L in range(self.num_layers):
                    assert len(loc_base[r][L]) == self.num_cache_entries  # K、V

            # ========= 判定是 DOWN 还是 UP/EQUAL =========
            tp_multiplier = self._tp_size[engine_id] // self._tp_size[self.engine_id]
            logger.info("[ADD] tp_multiplier=%s (dst_tp/src_tp = %s/%s)",
                        tp_multiplier, self._tp_size[engine_id], self._tp_size[self.engine_id])

            # -------------------------- Down 路径（prefill -> decode）--------------------------
            if tp_multiplier == 0 and not self._is_mla:
                # === 1) 基本几何参数（与原逻辑一致） ===
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

                # decode 端以“token 粒度”的单位数
                self.dst_num_blocks[engine_id] = num_blocks * B

                logger.info(
                    "[ADD][DOWN] group_size=%d remote_rank=%d peer_idx=%d token_len_local=%d token_len_total=%d full_len=%d seg_len=%d peer_off_tok=%d",
                    group_size, remote_rank, peer_idx, token_len_local, token_len_total, full_len, seg_len,
                    peer_off_tok)

                # === 2) 解析 decode pool，并解析/确定 base(engine_id) ===
                # 2.1 decode pool：优先 NIXL_DECODE_POOL 环境变量（如 "0,1"），否则沿用内部默认/推断
                if not hasattr(self, "_decode_pool") or self._decode_pool is None:
                    env_pool = os.getenv("NIXL_DECODE_POOL", "").strip()
                    if env_pool:
                        try:
                            self._decode_pool = [int(x) for x in env_pool.split(",") if x != ""]
                        except Exception:
                            logger.warning("Failed to parse NIXL_DECODE_POOL=%r, fallback to [0,1]", env_pool)
                            self._decode_pool = [0, 1]
                    else:
                        # 若环境没给，保守预设 2 张卡；你也可以按需改成其他默认
                        self._decode_pool = [0, 1]
                decode_pool = self._decode_pool

                # 2.2 如果远端元数据提供了 dev_ids，就直接据实映射；否则做一次“探测式”分配以锁定 base(engine_id)
                if not hasattr(self, "_engine_decode_base"):
                    self._engine_decode_base = {}
                base_fixed = engine_id in self._engine_decode_base

                def _candidate_bases(pool, need_len):
                    if need_len <= 0:
                        return []
                    if need_len == 1:
                        return list(range(len(pool)))
                    return [i for i in range(0, len(pool) - need_len + 1)]

                def _try_probe_base(base_idx: int) -> bool:
                    """用一个极小的远端 dlist 做 prep，能成功就说明 dev_id 对上了。"""
                    try:
                        r = remote_rank if remote_rank < agent_tp else 0
                        # 取一小段远端地址做试探：第 0 层、K 项、block 0
                        rbase_addr0 = self.kv_caches_base_addr[engine_id][r][0][0]
                        remote_dev = decode_pool[base_idx + r]
                        test_len = min(self.block_len, 4096)
                        test_desc = self.nixl_wrapper.get_xfer_descs([(rbase_addr0, test_len, int(remote_dev))], "VRAM")
                        # 只在远端侧准备（不会真正传输）
                        _h = self.nixl_wrapper.prep_xfer_dlist(self._remote_agents[engine_id][r], test_desc)
                        # 能走到这里说明这个 base 是匹配的
                        try:
                            _h.release()
                        except Exception:
                            pass
                        return True
                    except Exception:
                        return False

                if kv_caches_dev_ids is not None:
                    # 真实 dev id 从元数据来：对 rank r 取第一层、第一 entry 的 dev id 作为基准
                    # kv_caches_dev_ids 结构与 base_addr 对齐：[r][layer][entry]
                    first_dev = int(kv_caches_dev_ids[remote_rank][0][0])
                    # 反推出 base：decode_pool[base] == first_dev
                    try:
                        base = decode_pool.index(first_dev) - remote_rank
                    except ValueError:
                        base = None
                    if base is None or base < 0 or (base + agent_tp) > len(decode_pool):
                        raise RuntimeError(
                            f"[ADD][DOWN] invalid dev_ids mapping: first_dev={first_dev} pool={decode_pool} "
                            f"engine={engine_id} remote_rank={remote_rank} agent_tp={agent_tp}")
                    self._engine_decode_base[engine_id] = base
                    base_fixed = True
                elif not base_fixed:
                    # 无 dev_ids：探测锁定 base
                    ok = False
                    for cand in _candidate_bases(decode_pool, agent_tp):
                        if _try_probe_base(cand):
                            self._engine_decode_base[engine_id] = cand
                            ok = True
                            break
                    if not ok:
                        raise RuntimeError(f"[ADD][DOWN] cannot resolve decode base for engine={engine_id} "
                                           f"agent_tp={agent_tp} pool={decode_pool}; metadata has no dev_ids")
                base = self._engine_decode_base[engine_id]
                # 记录一下（gpu_id 打印首个槽位，便于和你的日志对齐）
                logger.info("[ADD][MAP] engine=%s role=VLLMWORKER -> gpu_id=%s (pool=%s)",
                            engine_id, decode_pool[base], decode_pool)

                # === 3) 构建本地 token 粒度 src dlist（与原逻辑一致） ===
                if 1 not in self.src_xfer_side_handles:
                    self.src_xfer_side_handles[1] = None
                local_dev_id = int(torch.cuda.current_device())
                src_blocks = []
                for layer in range(self.num_layers):
                    for base_addr in self.kv_caches_base_addr[self.engine_id][layer]:  # K、V
                        for bid in range(self.num_blocks):
                            base_block = base_addr + bid * seg_len
                            for t in range(B):
                                src_blocks.append((base_block + t * token_len_local, token_len_local, local_dev_id))
                logger.debug("[ADD][DOWN] src_blocks(token) count=%d", len(src_blocks))
                src_desc = self.nixl_wrapper.get_xfer_descs(src_blocks, "VRAM")
                self.src_xfer_side_handles[1] = self.nixl_wrapper.prep_xfer_dlist("", src_desc)

                # === 4) 构建远端 token 粒度 dst dlist：**关键改动：remote_dev = decode_pool[base + r]** ===
                if engine_id not in self.dst_xfer_side_handles:
                    self.dst_xfer_side_handles[engine_id] = {}
                dst_blocks = []
                remote_dev_id = int(decode_pool[base + remote_rank])
                for layer in range(self.num_layers):
                    layer_bases = self.kv_caches_base_addr[engine_id][remote_rank][layer]  # [K_base, V_base]
                    for entry_idx, rbase in enumerate(layer_bases):
                        for bid in range(num_blocks):
                            base_block = rbase + bid * full_len
                            for t in range(B):
                                dst_blocks.append((base_block + t * token_len_total + peer_off_tok,
                                                   token_len_local, remote_dev_id))
                logger.debug("[ADD][DOWN] dst_blocks(token) count=%d remote_rank=%d mapped_dev=%d",
                             len(dst_blocks), remote_rank, remote_dev_id)
                dst_desc = self.nixl_wrapper.get_xfer_descs(dst_blocks, "VRAM")
                self.dst_xfer_side_handles[engine_id][remote_rank] = self.nixl_wrapper.prep_xfer_dlist(
                    self._remote_agents[engine_id][remote_rank], dst_desc
                )

                # 懒连接
                try:
                    self.nixl_wrapper.make_connection(self._remote_agents[engine_id][remote_rank])
                except Exception as e:
                    logger.debug("make_connection(%s) lazy: %s", self._remote_agents[engine_id][remote_rank], e)

                # === 5) READ-DOWN：本地标准块布局 & 远端块布局（同样用 mapped_dev） ===
                if "read_down_src" not in self.src_xfer_side_handles:
                    blocks_local = []
                    local_dev_id = int(torch.cuda.current_device())
                    for layer in range(self.num_layers):
                        for base_addr in self.kv_caches_base_addr[self.engine_id][layer]:  # K、V
                            for bid in range(self.num_blocks):
                                block_offset = bid * self.block_len
                                blocks_local.append((base_addr + block_offset, self.block_len, local_dev_id))
                    logger.debug("[ADD][DOWN] read_down_src blocks(count)=%d", len(blocks_local))
                    descs_local = self.nixl_wrapper.get_xfer_descs(blocks_local, "VRAM")
                    self.src_xfer_side_handles["read_down_src"] = self.nixl_wrapper.prep_xfer_dlist("", descs_local)

                blocks_remote = []
                remote_dev_id_rd = int(decode_pool[base + remote_rank])
                for layer in range(self.num_layers):
                    layer_bases = self.kv_caches_base_addr[engine_id][remote_rank][layer]
                    for entry_idx, rbase in enumerate(layer_bases):
                        for bid in range(num_blocks):
                            block_offset = bid * self.block_len
                            blocks_remote.append((rbase + block_offset, self.block_len, remote_dev_id_rd))
                logger.debug("[ADD][DOWN] read_down_dst blocks(count)=%d remote_rank=%d mapped_dev=%d",
                             len(blocks_remote), remote_rank, remote_dev_id_rd)

                descs_remote = self.nixl_wrapper.get_xfer_descs(blocks_remote, "VRAM")
                self.dst_xfer_side_handles[engine_id]["read_down_dst"] = self.nixl_wrapper.prep_xfer_dlist(
                    self._remote_agents[engine_id][remote_rank], descs_remote
                )

                if not hasattr(self, "dst_num_blocks_read"):
                    self.dst_num_blocks_read = {}
                self.dst_num_blocks_read[engine_id] = num_blocks

                # 预拨号一次
                try:
                    self.nixl_wrapper.make_connection(self._remote_agents[engine_id][remote_rank])
                except Exception as e:
                    logger.debug("[ADD][READ-DOWN] make_connection lazy: %s", e)

                logger.info("[ADD][DOWN][READY] engine=%s local_rank=%d remote_rank=%d "
                            "src_keys=%s dst_keys=%s dst_units(token)=%s read_down_keys=%s",
                            engine_id, self.rank, remote_rank,
                            list(self.src_xfer_side_handles.keys()),
                            list(self.dst_xfer_side_handles[engine_id].keys()),
                            self.dst_num_blocks[engine_id],
                            list(self.dst_xfer_side_handles[engine_id].keys()))
                return agent_names

            # -------------------------- Up/Equal 路径（保持官方/你现状）--------------------------
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
                                # 注意：Up 路径沿用原来的 self.rank 作为设备号，不做任何改动
                                blocks_data.append((base_addr + block_offset + tp_off, dst_block_len, self.rank))
                logger.debug("[ADD][UP] src_blocks(count)=%d key=%d", len(blocks_data), tp_multiplier)
                descs = self.nixl_wrapper.get_xfer_descs(blocks_data, "VRAM")
                self.src_xfer_side_handles[tp_multiplier] = self.nixl_wrapper.prep_xfer_dlist("", descs)

            self.dst_num_blocks[engine_id] = num_blocks
            for i in range(tp_multiplier):
                blocks_data = []
                remote_idx = self.rank * tp_multiplier + i  # 官方连续映射
                for layer_id in range(self.num_layers):
                    layer_bases = loc_base[remote_idx][layer_id]
                    for entry_idx, base_addr in enumerate(layer_bases):
                        for block_id in range(num_blocks):
                            block_offset = block_id * dst_block_len
                            blocks_data.append((base_addr + block_offset, dst_block_len, int(remote_idx)))
                logger.debug("[ADD][UP] dst_blocks(count)=%d i=%d remote_idx=%d", len(blocks_data), i, remote_idx)
                descs = self.nixl_wrapper.get_xfer_descs(blocks_data, "VRAM")
                self.dst_xfer_side_handles[engine_id][i] = self.nixl_wrapper.prep_xfer_dlist(
                    self._remote_agents[engine_id][remote_idx], descs
                )
                try:
                    self.nixl_wrapper.make_connection(self._remote_agents[engine_id][remote_idx])
                except Exception as e:
                    logger.debug("[ADD] make_connection(%s) lazy: %s", self._remote_agents[engine_id][remote_idx], e)

            logger.info("[ADD][UP][READY] engine=%s local_rank=%d src_keys=%s dst_keys=%s dst_num_blocks=%s",
                        engine_id, self.rank,
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
