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

from .kv_rearrange import rearrange_tensors
from contextlib import contextmanager
from functools import lru_cache

logger = init_logger(__name__)

# ---------- A3: 二级 LRU 缓存的块->token 展开实现（按连续段 + block_size 做 key） ----------
@lru_cache(maxsize=4096)
def _expand_blocks_to_tokens_cached_key(ranges_key: tuple, B: int) -> Tuple[int, ...]:
    """
    将若干 [a,b] 的块区间展开为 token 索引（连续），并缓存。
    注意把 block_size(B) 纳入 key，避免不同 B 造成错误复用。
    """
    if not ranges_key or B <= 0:
        return tuple()
    out = []
    # 对于连续块 [a,b]，token 索引是 [a*B, (b+1)*B)
    for a, b in ranges_key:
        start = int(a) * B
        end = (int(b) + 1) * B
        out.extend(range(start, end))
    return tuple(out)


# Lazy import nixl_wrapper to avoid loading nixl_bindings if nixl is not used
try:
    from nixl._api import nixl_agent as NixlWrapper
    logger.info("NIXL is available")
except ImportError:
    logger.warning("NIXL is not available")
    NixlWrapper = None


def _stable_hash_u32(s: str) -> int:
    # 稳定一致的 32-bit 哈希（跨进程/多机一致）
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16)


def _pick_from_pool(pool: list[int], base_slot: int, worker_idx: int) -> int:
    # 空池时回落到当前 CUDA 设备（保底）
    if not pool:
        try:
            return int(torch.cuda.current_device())
        except Exception:
            return 0
    # 以 base_slot 为起点，worker_idx 连续展开：0->pool[0], 1->pool[1], ...
    return int(pool[(base_slot + worker_idx) % len(pool)])


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
        self.block_size = None
        self.head_dim = None
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

        self._downscale_info = {}

        # A4: (层,entry) 列表缓存
        self._le_list_cache = None

        # ---- engine_id fingerprinting (strict check) ----
        self._engine_fingerprint = {}  # engine_id -> fp
        self._engine_agent_tp = {}     # engine_id -> tp size for check

        # engine 是否做过“重型”准备（注册远端 agents 等）
        self._engine_prepped = {}

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

    # ---------- A3: 二级 LRU：以连续段为 key 的展开 ----------
    def _ranges_key(self, block_ids: List[int]) -> tuple:
        rngs = self._get_ranges(block_ids)
        return tuple((int(a), int(b)) for a, b in rngs)

    def _expand_blocks_to_tokens_cached(self, block_ids: List[int]) -> List[int]:
        """优先使用二级 LRU 缓存的块->token 展开。"""
        if not block_ids:
            return []
        B = int(self.block_size)
        rk = self._ranges_key(block_ids)
        return list(_expand_blocks_to_tokens_cached_key(rk, B))

    # ---------- A4: 缓存 (层,entry) 列表 ----------
    def _get_le_list(self):
        if self._le_list_cache is None:
            self._le_list_cache = [(L, E) for L in range(self.num_layers) for E in range(self.num_cache_entries)]
        return self._le_list_cache

    def _write_blocks_down(self, local_block_ids, remote_block_ids, dst_engine_id, notify_msg):
        """
        DOWN：组长聚合 + 少 IOV 一次下发（默认开启）
          - NIXL_DOWN_GATHER=1（默认）且 group_size>1 时启用；
          - 否则回退到本文件里你之前的“单次 submit（token 粒度预制 dlist）”实现。
        """

        # --------- 小工具（仅在本函数内）---------
        def _ranges(ids):
            if not ids: return []
            out, s, p = [], ids[0], ids[0]
            for x in ids[1:]:
                if x == p + 1:
                    p = x
                else:
                    out.append((s, p));
                    s = p = x
            out.append((s, p))
            return out

        def _pack_local_std(L: int, entry_idx: int, ranges):
            # kv_caches[L][entry] 形状（非MLA）：[num_blocks, B, H_loc, C]
            src = self.kv_caches[L][entry_idx]
            parts = [src[a:b + 1].contiguous() for (a, b) in ranges]
            return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0).contiguous()

        def _persist_meta(dst_engine_id: str, key: str, group_size: int, peer_idx: int, meta: dict):
            import os, msgspec, time
            base = os.getenv("NIXL_BARRIER_DIR", "/dev/shm" if os.path.isdir("/dev/shm") else "/tmp")

            def _safe(s, n):
                s = str(s);
                r = []
                for ch in s:
                    if ch.isalnum() or ch in ("-", "_"): r.append(ch)
                    if len(r) >= n: break
                return "".join(r) or "k"

            d = os.path.join(base, f"nixl_gather_{_safe(dst_engine_id, 16)}_{_safe(key, 24)}_{group_size}")
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"{peer_idx}.msgpack")
            b = msgspec.msgpack.encode(meta)
            tmp = f"{path}.tmp.{os.getpid()}"
            with open(tmp, "wb") as f:
                f.write(b)
            os.replace(tmp, path)

        def _load_all_meta(dst_engine_id: str, key: str, group_size: int, timeout_ms: int = 200):
            import os, msgspec, time
            base = os.getenv("NIXL_BARRIER_DIR", "/dev/shm" if os.path.isdir("/dev/shm") else "/tmp")

            def _safe(s, n):
                s = str(s);
                r = []
                for ch in s:
                    if ch.isalnum() or ch in ("-", "_"): r.append(ch)
                    if len(r) >= n: break
                return "".join(r) or "k"

            d = os.path.join(base, f"nixl_gather_{_safe(dst_engine_id, 16)}_{_safe(key, 24)}_{group_size}")
            metas = []
            deadline = time.time() + timeout_ms / 1000.0
            for i in range(group_size):
                p = os.path.join(d, f"{i}.msgpack")
                while not os.path.exists(p) and time.time() < deadline:
                    time.sleep(0.001)
                if not os.path.exists(p):
                    raise RuntimeError(f"[DOWN-GATHER] missing meta for peer {i}")
                with open(p, "rb") as f:
                    metas.append(msgspec.msgpack.decode(f.read()))
            return metas

        def _remote_pool_index(engine_id: str, r_idx: int, layer: int, entry_idx: int) -> int:
            # 先看 kv_caches_dev_ids
            devs = self.kv_caches_dev_ids.get(engine_id)
            if devs is not None:
                try:
                    return int(devs[r_idx][layer][entry_idx])
                except Exception:
                    pass
            # 再看映射
            s = os.getenv("NIXL_MAP_PREFILLWORKER", "").strip()
            if s:
                m = {}
                for kv in s.split(","):
                    kv = kv.strip()
                    if not kv: continue
                    a, b = (kv.split("->", 1) if "->" in kv else kv.split(":", 1))
                    try:
                        m[int(a.strip())] = int(b.strip())
                    except:
                        pass
                if r_idx in m: return int(m[r_idx])
            return 0

        def _leader_merge(per_peer_K, per_peer_V, ngroups: int):
            """把 per-peer 的 [N,B,H_loc,C] 按组拼成长条 grouped，再用 Triton 合并到 standard。"""
            num_layers = len(per_peer_K[0])
            N, B, H_loc, C = per_peer_K[0][0].shape
            H_tot = H_loc * ngroups
            K_std_layers, V_std_layers = [], []

            # 组内顺序：允许通过 _down_peer_perm 自定义；否则 0..ng-1
            perm = self._down_peer_perm(ngroups) if hasattr(self, "_down_peer_perm") else list(range(ngroups))
            seg_elems = N * B * H_loc * C

            for L in range(num_layers):
                K_grp = torch.empty((N, B, H_tot, C), dtype=per_peer_K[0][L].dtype, device=per_peer_K[0][L].device)
                V_grp = torch.empty_like(K_grp)
                Kdst = K_grp.view(-1);
                Vdst = V_grp.view(-1)
                for g, real in enumerate(perm):
                    Ksrc = per_peer_K[real][L].contiguous().view(-1)
                    Vsrc = per_peer_V[real][L].contiguous().view(-1)
                    Kdst[g * seg_elems:(g + 1) * seg_elems].copy_(Ksrc)
                    Vdst[g * seg_elems:(g + 1) * seg_elems].copy_(Vsrc)
                with self._timing.span("write_down.leader.kernel_merge"):
                    K_std_layers.append(_grouped_to_standard(K_grp, ngroups))
                    V_std_layers.append(_grouped_to_standard(V_grp, ngroups))
            return K_std_layers, V_std_layers

        def _build_descs_src(base_ptr: int, dev_id: int, bytes_per_block: int, ranges):
            descs = []
            off = 0
            for a, b in ranges:
                n = b - a + 1
                descs.append((base_ptr + off, n * bytes_per_block, dev_id))
                off += n * bytes_per_block
            return descs

        def _build_descs_dst(base_ptr: int, dev_id: int, bytes_per_block: int, ranges):
            return [(base_ptr + a * bytes_per_block, (b - a + 1) * bytes_per_block, dev_id) for (a, b) in ranges]

        # ========= 正式逻辑 =========
        with self._timing.span("write_down"):
            info = self._downscale_info[dst_engine_id]
            assert info is not None, "[WRITE-DOWN] downscale info missing"
            remote_rank = info["remote_rank"]
            is_leader = bool(info.get("notify_leader"))
            notify_key = notify_msg if isinstance(notify_msg, str) else str(notify_msg)

            # 句柄健全性
            if 1 not in self.src_xfer_side_handles or self.src_xfer_side_handles[1] is None:
                raise RuntimeError(f"[WRITE-DOWN] missing src token handle (rank={self.rank})")
            if (dst_engine_id not in self.dst_xfer_side_handles or
                    remote_rank not in self.dst_xfer_side_handles[dst_engine_id] or
                    self.dst_xfer_side_handles[dst_engine_id][remote_rank] is None):
                raise RuntimeError(f"[WRITE-DOWN] missing dst token handle (rank={self.rank} rr={remote_rank})")

            loc_ranges = _ranges(local_block_ids)
            rem_ranges = _ranges(remote_block_ids)
            N_blocks = sum(b - a + 1 for a, b in loc_ranges)
            if N_blocks == 0:
                if is_leader and notify_key:
                    self.nixl_wrapper.send_notif(self._remote_agents[dst_engine_id][remote_rank], notify_key)
                return

            # 是否启用“组长聚合”
            ngroups = int(info["group_size"])
            use_gather = (os.getenv("NIXL_DOWN_GATHER", "1") != "0") and (ngroups > 1)

            if use_gather:
                # ========== 每个 peer 本机：打包 [N,B,H_loc,C] 到连续显存 + 写元数据 ==========
                with self._timing.span("write_down.pack_local"):
                    per_layer_K = [_pack_local_std(L, 0, loc_ranges) for L in range(self.num_layers)]
                    per_layer_V = [_pack_local_std(L, 1, loc_ranges) for L in range(self.num_layers)]
                    md = self.get_agent_metadata()
                    meta = {
                        "agent_md": bytes(md),
                        "device": int(per_layer_K[0].get_device()),
                        "dtype": str(per_layer_K[0].dtype),
                        "N": int(N_blocks),
                        "B": int(self.block_size),
                        "H_loc": int(self.num_heads),
                        "C": int(self.head_dim),
                        "layers": [
                            {
                                "K_addr": int(per_layer_K[L].data_ptr()),
                                "K_len": int(per_layer_K[L].numel() * per_layer_K[L].element_size()),
                                "K_dev": int(per_layer_K[L].get_device()),
                                "V_addr": int(per_layer_V[L].data_ptr()),
                                "V_len": int(per_layer_V[L].numel() * per_layer_V[L].element_size()),
                                "V_dev": int(per_layer_V[L].get_device()),
                            }
                            for L in range(self.num_layers)
                        ],
                    }
                    _persist_meta(dst_engine_id, notify_key, ngroups, int(info["peer_idx"]), meta)

                # 组内 barrier，保证所有 peers 已写 meta
                with self._timing.span("write_down.barrier"):
                    self._barrier_mark_and_wait(dst_engine_id, notify_key, ngroups, info["peer_idx"], is_leader)

                # 非 leader：到此结束（不与 decode 直接通信）
                if not is_leader:
                    return

                # ========== leader：读取所有 meta，拉取其他 peer 的大连片 ==========
                with self._timing.span("write_down.leader.load_meta"):
                    metas = _load_all_meta(dst_engine_id, notify_key, ngroups)

                per_peer_K = [[None] * self.num_layers for _ in range(ngroups)]
                per_peer_V = [[None] * self.num_layers for _ in range(ngroups)]
                my_peer = int(info["peer_idx"])
                for L in range(self.num_layers):
                    per_peer_K[my_peer][L] = per_layer_K[L]
                    per_peer_V[my_peer][L] = per_layer_V[L]

                BACKENDS = ["UCX"] if os.getenv("NIXL_FORCE_UCX", "1") == "1" else None
                inflight = []
                # 拉取其他 peers（READ）
                for g, m in enumerate(metas):
                    if g == my_peer:
                        continue
                    peer_agent = self.nixl_wrapper.add_remote_agent(m["agent_md"])
                    for L in range(self.num_layers):
                        # 目标：leader 本机 [N,B,H_loc,C]
                        N = int(m["N"]);
                        B = int(m["B"]);
                        H_loc = int(m["H_loc"]);
                        C = int(m["C"])
                        K_dst = torch.empty((N, B, H_loc, C), dtype=per_layer_K[0].dtype, device=per_layer_K[0].device)
                        V_dst = torch.empty_like(K_dst)
                        per_peer_K[g][L] = K_dst;
                        per_peer_V[g][L] = V_dst

                        # local dlist（dst）
                        K_loc_desc = self.nixl_wrapper.get_xfer_descs(
                            [(int(K_dst.data_ptr()), int(K_dst.numel() * K_dst.element_size()),
                              int(K_dst.get_device()))], "VRAM")
                        V_loc_desc = self.nixl_wrapper.get_xfer_descs(
                            [(int(V_dst.data_ptr()), int(V_dst.numel() * V_dst.element_size()),
                              int(V_dst.get_device()))], "VRAM")
                        K_loc_h = self.nixl_wrapper.prep_xfer_dlist("", K_loc_desc, backends=BACKENDS)
                        V_loc_h = self.nixl_wrapper.prep_xfer_dlist("", V_loc_desc, backends=BACKENDS)
                        # remote dlist（src）
                        row = m["layers"][L]
                        K_rem_desc = self.nixl_wrapper.get_xfer_descs(
                            [(int(row["K_addr"]), int(row["K_len"]), int(row["K_dev"]))], "VRAM")
                        V_rem_desc = self.nixl_wrapper.get_xfer_descs(
                            [(int(row["V_addr"]), int(row["V_len"]), int(row["V_dev"]))], "VRAM")
                        K_rem_h = self.nixl_wrapper.prep_xfer_dlist(peer_agent, K_rem_desc, backends=BACKENDS)
                        V_rem_h = self.nixl_wrapper.prep_xfer_dlist(peer_agent, V_rem_desc, backends=BACKENDS)
                        # READ
                        hK = self.nixl_wrapper.make_prepped_xfer("READ", K_loc_h, [0], K_rem_h, [0], b"",
                                                                 backends=BACKENDS)
                        hV = self.nixl_wrapper.make_prepped_xfer("READ", V_loc_h, [0], V_rem_h, [0], b"",
                                                                 backends=BACKENDS)
                        self.nixl_wrapper.transfer(hK);
                        inflight.append(hK)
                        self.nixl_wrapper.transfer(hV);
                        inflight.append(hV)

                if inflight:
                    with self._timing.span("write_down.leader.wait_gather"):
                        self._wait_many(inflight)
                        inflight.clear()

                # ========== leader：Triton 合并到 standard（H_tot = H_loc*ngroups） ==========
                with self._timing.span("write_down.leader.merge"):
                    K_std_layers, V_std_layers = _leader_merge(per_peer_K, per_peer_V, ngroups)

                # ========== leader：按连续段 少 IOV 一次写到 decode ==========
                with self._timing.span("write_down.leader.write_decode"):
                    B = int(self.block_size)
                    H_loc = int(self.num_heads)
                    C = int(self.head_dim)
                    e = self.kv_caches[0][0].element_size()
                    H_tot = H_loc * ngroups
                    token_len_total = H_tot * C * e
                    bytes_per_block = B * token_len_total

                    rr = remote_rank
                    dst_agent = self._remote_agents[dst_engine_id][rr]
                    for L in range(self.num_layers):
                        for entry_idx, std in ((0, K_std_layers[L]), (1, V_std_layers[L])):
                            src_ptr = int(std.data_ptr());
                            src_dev = int(std.get_device())
                            src_desc = _build_descs_src(src_ptr, src_dev, bytes_per_block, loc_ranges)
                            src_xfer = self.nixl_wrapper.get_xfer_descs(src_desc, "VRAM")
                            src_h = self.nixl_wrapper.prep_xfer_dlist("", src_xfer, backends=BACKENDS)

                            base_decode = int(self.kv_caches_base_addr[dst_engine_id][rr][L][entry_idx])
                            dst_dev = _remote_pool_index(dst_engine_id, rr, L, entry_idx)
                            dst_desc = _build_descs_dst(base_decode, int(dst_dev), bytes_per_block, rem_ranges)
                            dst_xfer = self.nixl_wrapper.get_xfer_descs(dst_desc, "VRAM")
                            dst_h = self.nixl_wrapper.prep_xfer_dlist(dst_agent, dst_xfer, backends=BACKENDS)

                            idx = list(range(len(src_desc)))
                            h = self.nixl_wrapper.make_prepped_xfer(
                                "WRITE",
                                src_h, idx,
                                dst_h, idx,
                                (notify_key if (entry_idx == self.num_cache_entries - 1) else b""),
                                backends=BACKENDS,
                                skip_desc_merge=True,
                            )
                            self.nixl_wrapper.transfer(h)
                            self._wait_many([h])
                return

            # ========== 回退路径：你原来的“单次 submit”实现 ==========
            # ---- block -> token 连续段 ----
            def _to_token_ranges(block_ids: list[int]) -> list[tuple[int, int]]:
                if not block_ids: return []
                B = int(self.block_size)
                out = []
                for a, b in _ranges(block_ids):
                    out.append((a * B, (b + 1) * B))
                return out

            with self._timing.span("write_down.expand_tokens"):
                loc_tok = _to_token_ranges(local_block_ids)
                rem_tok = _to_token_ranges(remote_block_ids)

            Ntok = sum(hi - lo for lo, hi in loc_tok)
            notify_payload = notify_msg if isinstance(notify_msg, str) else str(notify_msg)
            if Ntok == 0:
                if bool(info.get("notify_leader")) and notify_payload:
                    self.nixl_wrapper.send_notif(self._remote_agents[dst_engine_id][remote_rank], notify_payload)
                return

            per_entry_src = int(self.num_blocks) * int(self.block_size)
            per_entry_dst = int(self.dst_num_blocks[dst_engine_id])
            same_layout = (per_entry_src == per_entry_dst and len(loc_tok) == len(rem_tok)
                           and all((b1 - a1) == (b2 - a2) for (a1, b1), (a2, b2) in zip(loc_tok, rem_tok)))

            if hasattr(self, "_get_le_list"):
                le_list = self._get_le_list()
            else:
                le_list = [(L, E) for L in range(self.num_layers) for E in range(self.num_cache_entries)]

            BACKENDS = ["UCX"] if os.getenv("NIXL_FORCE_UCX", "1") == "1" else None

            with self._timing.span("write_down.barrier"):
                self._barrier_mark_and_wait(dst_engine_id, notify_payload, info["group_size"], info["peer_idx"],
                                            bool(info.get("notify_leader")))

            with self._timing.span("write_down.build_idx"):
                local_idx = [];
                remote_idx = None if same_layout else []
                _lex = local_idx.extend;
                _rex = (remote_idx.extend if remote_idx is not None else None)
                for (layer, entry) in le_list:
                    base_layer_src = layer * (self.num_cache_entries * per_entry_src)
                    base_layer_dst = layer * (self.num_cache_entries * per_entry_dst)
                    base_entry_src = base_layer_src + entry * per_entry_src
                    base_entry_dst = base_layer_dst + entry * per_entry_dst
                    for (lo, hi) in loc_tok: _lex(range(base_entry_src + lo, base_entry_src + hi))
                    if remote_idx is not None:
                        for (lo, hi) in rem_tok: _rex(range(base_entry_dst + lo, base_entry_dst + hi))

            with self._timing.span("write_down.submit"):
                src_hdl = self.src_xfer_side_handles[1]
                dst_hdl = self.dst_xfer_side_handles[dst_engine_id][remote_rank]
                h = self.nixl_wrapper.make_prepped_xfer(
                    "WRITE",
                    src_hdl, local_idx,
                    dst_hdl, (local_idx if same_layout else remote_idx),
                    (notify_payload if bool(info.get("notify_leader")) else ""),
                    backends=BACKENDS
                )
                self.nixl_wrapper.transfer(h)

            with self._timing.span("write_down.wait_window"):
                self._wait_many([h])

            LxE = self.num_layers * self.num_cache_entries
            total_iov = len(local_idx)
            import logging;
            logging.getLogger(__name__).info(
                "[WRITE][DOWN-SINGLE] submit_once=True same_layout=%s LxE=%d Ntok=%d total_iov=%d",
                same_layout, LxE, Ntok, total_iov
            )

    def _read_blocks_down(self, local_block_ids, staging_block_ids, remote_block_ids, dst_engine_id):
        """
        Prefill<--READ--Decode（Downscale）路径：
        - 采用 UCX 后端（可通过 NIXL_FORCE_UCX=0 关闭）
        - 大 dlist 分块 + 有限并发窗口，降低控制面时延与 agent 校验成本
        - 读完在本机做 standard->grouped 的 GPU 重排；支持按块再切小片降低显存峰值
          （通过 NIXL_READ_REARRANGE_CHUNK_BLOCKS 控制）
        """

        with self._timing.span("read_down"):
            down = self._downscale_info.get(dst_engine_id)
            assert down is not None, "[READ-DOWN] downscale info missing"

            # ===== 句柄就绪性强校验 =====
            if ("read_down_src" not in self.src_xfer_side_handles
                    or self.src_xfer_side_handles["read_down_src"] is None):
                raise RuntimeError(
                    f"[READ-DOWN] missing local READ dst (staging) handle: "
                    f"engine={self.engine_id} local_rank={self.rank} keys={list(self.src_xfer_side_handles.keys())}"
                )
            if (dst_engine_id not in self.dst_xfer_side_handles
                    or "read_down_dst" not in self.dst_xfer_side_handles[dst_engine_id]
                    or self.dst_xfer_side_handles[dst_engine_id]["read_down_dst"] is None):
                raise RuntimeError(
                    f"[READ-DOWN] missing remote READ src handle: "
                    f"dst_engine={dst_engine_id} local_rank={self.rank} "
                    f"dst_keys_top={list(self.dst_xfer_side_handles.keys())} "
                    f"dst_keys_inner={list(self.dst_xfer_side_handles.get(dst_engine_id, {}).keys())}"
                )

            dst_handle = self.src_xfer_side_handles["read_down_src"]  # 本地 staging（标准布局）作为 READ 目的地
            src_handle = self.dst_xfer_side_handles[dst_engine_id]["read_down_dst"]  # 远端 decode 作为 READ 来源（块粒度）

            # ===== 把“块 id”展开成 dlist 描述 id（顺序：layer → entry(K/V) → block）=====
            def _ids_blockwise(num_blocks_total: int, block_ids: list[int]) -> list[int]:
                ids = []
                for layer in range(self.num_layers):
                    for entry in range(self.num_cache_entries):  # K、V
                        for b in block_ids:
                            ids.append(layer * self.num_cache_entries * num_blocks_total
                                       + entry * num_blocks_total + int(b))
                return ids

            # 远端的“块总数”在 DOWN 模式下就是对端 decode 的 num_blocks（add_remote_agent 里赋过）
            num_blocks_remote = int(self.dst_num_blocks_read[dst_engine_id])
            src_desc_ids = _ids_blockwise(num_blocks_remote, remote_block_ids)
            dst_desc_ids = _ids_blockwise(self.num_blocks, staging_block_ids)
            if len(src_desc_ids) != len(dst_desc_ids):
                raise RuntimeError(f"[READ-DOWN] desc len mismatch: src={len(src_desc_ids)} dst={len(dst_desc_ids)}")

            # ===== UCX+分块流水传输 =====
            MAX_IOV_RD = int(os.getenv("NIXL_MAX_IOV_READ", "16384"))  # 每个请求最多多少个 desc
            MAX_INFLIGHT_RD = int(os.getenv("NIXL_MAX_INFLIGHT_READ", "8"))  # 并发请求窗口
            BACKENDS = ["UCX"] if os.getenv("NIXL_FORCE_UCX", "1") == "1" else None

            inflight = []
            total_reqs = 0
            for off in range(0, len(dst_desc_ids), MAX_IOV_RD):
                lo, hi = off, min(off + MAX_IOV_RD, len(dst_desc_ids))
                h = self.nixl_wrapper.make_prepped_xfer(
                    "READ",
                    dst_handle, dst_desc_ids[lo:hi],
                    src_handle, src_desc_ids[lo:hi],
                    "",
                    backends=BACKENDS,
                )
                self.nixl_wrapper.transfer(h)
                inflight.append(h)
                total_reqs += 1
                if len(inflight) >= MAX_INFLIGHT_RD:
                    self._wait_many(inflight)
                    inflight.clear()
            if inflight:
                self._wait_many(inflight)
                inflight.clear()
            logger.info("[READ-DOWN] chunked_reqs=%d iov_per_req<=%d inflight<=%d",
                        total_reqs, MAX_IOV_RD, MAX_INFLIGHT_RD)

            # ===== 重排：standard -> grouped =====
            # ngroups = tp_prefill // tp_decode；更稳妥地直接用 add_remote_agent 填的 group_size
            ngroups = int(down.get("group_size", 1))
            if ngroups <= 1 or self._is_mla:
                return

            # H 必须整除 ngroups；不满足则跳过（避免非法访存）
            try:
                # 形状: [N_blocks, B, H, C]
                sample = self.kv_caches[0][0]
                H = int(sample.shape[2])
            except Exception:
                H = None
            if not H or (H % ngroups != 0):
                logger.warning("[READ-DOWN] skip rearrange: invalid H=%s for ngroups=%s (must divide).", H, ngroups)
                return

            # 将本次涉及的块区间做 1:1 重排（local 为目标 grouped，staging 为标准布局）
            local_ranges = self._get_ranges(local_block_ids)
            staging_ranges = self._get_ranges(staging_block_ids)
            if len(local_ranges) != len(staging_ranges):
                # 理论上两者等长；不等长则按“对齐后最短”处理
                n = min(len(local_ranges), len(staging_ranges))
                local_ranges, staging_ranges = local_ranges[:n], staging_ranges[:n]

            # 限制一次 contiguous 切片的块数，降低临时显存峰值
            CHUNK_BLKS = int(os.getenv("NIXL_READ_REARRANGE_CHUNK_BLOCKS", "16"))

            from .kv_rearrange import rearrange_tensors_read_down

            for (l0, l1), (s0, s1) in zip(local_ranges, staging_ranges):
                n_blocks = (l1 - l0 + 1)
                if CHUNK_BLKS > 0:
                    steps = range(l0, l1 + 1, CHUNK_BLKS)
                else:
                    steps = [l0]  # 单批

                for start in steps:
                    end = min(start + (CHUNK_BLKS if CHUNK_BLKS > 0 else n_blocks), l1 + 1)
                    # staging 区间起点跟随块偏移对齐
                    s_start = s0 + (start - l0)
                    s_end = s_start + (end - start)

                    # 切 [N(块数), B, H, C]，做成连续小片后调用 kernel，再写回
                    for kv_cache in self.kv_caches:  # (K, V) / [KV]
                        for cache in kv_cache:
                            t_std = cache[s_start:s_end].contiguous()  # staging（标准）
                            t_grp = cache[start:end].contiguous()  # 目标（组段）
                            rearrange_tensors_read_down(t_std, t_grp, ngroups)
                            cache[start:end].copy_(t_grp)

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

        # A4: 每次注册后，重置 (层,entry) 列表缓存
        self._le_list_cache = None

        if _env_flag("NIXL_PUBLISH_SELF_MD", True):
            try:
                self._persist_remote_md_cache(
                    engine_id=self.engine_id,
                    agent_metadata=self.get_agent_metadata(),
                    kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id],
                    num_blocks=int(self.num_blocks),
                    kv_caches_dev_ids=self.kv_caches_dev_ids.get(self.engine_id),
                    agent_tp=int(self._tp_size[self.engine_id]),
                )
                logger.info("[MD-CACHE][SELF] published engine=%s", self.engine_id)
            except Exception as e:
                logger.debug("[MD-CACHE][SELF] publish failed: %s", e)

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
        # A3: 清理二级 LRU 缓存，避免长跑时内存占用
        try:
            _expand_blocks_to_tokens_cached_key.cache_clear()
        except Exception:
            pass

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
            # ★ 关键：写入前先规范化，确保是 List[bytes]，避免被按 int 序列还原
            agent_metadata = self._coerce_agent_metadata(agent_metadata)

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
            logger.debug("[MD-CACHE] persisted for engine=%s path=%s size=%dB",
                         engine_id, path, len(b))
        except Exception as e:
            logger.debug("[MD-CACHE] persist failed: %s", e)

    def _coerce_agent_metadata(self, md) -> List[bytes]:
        """把各种可能的形式统一成 List[bytes]，避免把 bytes 当成可迭代的 int。"""
        if md is None:
            return []
        # 单段：bytes / bytearray / memoryview
        if isinstance(md, (bytes, bytearray, memoryview)):
            return [bytes(md)]
        # 列表/元组：逐个转成 bytes
        if isinstance(md, (list, tuple)):
            out = []
            for x in md:
                if isinstance(x, (bytes, bytearray, memoryview)):
                    out.append(bytes(x))
                else:
                    # 某些序列化器可能给成了 list[int]；兜底拼成 bytes
                    if isinstance(x, list) and all(isinstance(i, int) for i in x):
                        out.append(bytes(x))
                    else:
                        raise TypeError(f"agent_metadata elem must be bytes-like, got {type(x).__name__}")
            return out
        # 其它类型不接受（如 str 路径不在此通道使用）
        raise TypeError(f"agent_metadata must be bytes or list of bytes, got {type(md).__name__}")

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
            # ★ 关键：强制把 agent_metadata 变成 List[bytes]
            agent_metadata = self._coerce_agent_metadata(data.get("agent_metadata"))
            kv_rows_norm = self._normalize_kv_rows(engine_id, data["kv_caches_base_addr"], agent_tp)
            self.add_remote_agent(
                engine_id=data["engine_id"],
                agent_metadata=agent_metadata,
                agent_tp=agent_tp,
                kv_caches_base_addr=kv_rows_norm,
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

                assert len(staging_block_ids) == len(local_block_ids), \
                    f"[WRITE] len mismatch: staging={len(staging_block_ids)} local={len(local_block_ids)}"
                assert len(remote_block_ids) == len(local_block_ids), \
                    f"[WRITE] len mismatch: remote={len(remote_block_ids)} local={len(local_block_ids)}"

                # 只等待 add_remote_agent 的产物就绪；若本进程尚未注册该 dst，则尝试从共享缓存“收养”
                wait_ms = int(os.getenv("NIXL_READY_WAIT_MS", "3000"))
                t0 = time.time()
                last_missing = "unknown"

                while True:
                    # —— 若缺元数据/句柄，优先尝试从 NIXL_MD_CACHE_DIR 收养一次（调用 add_remote_agent）
                    if (dst_engine_id not in self._tp_size
                            or dst_engine_id not in self.dst_xfer_side_handles
                            or not self.dst_xfer_side_handles.get(dst_engine_id)):
                        try:
                            if self._adopt_remote_md_from_cache(dst_engine_id):
                                logger.info("[WRITE][ADOPT] adopted remote metadata for dst=%s", dst_engine_id)
                        except Exception as _e:
                            logger.debug("[WRITE][ADOPT] adopt failed for dst=%s: %s", dst_engine_id, _e)

                    down = self._downscale_info.get(dst_engine_id)
                    if down is not None:
                        rr = down.get("remote_rank")
                        src_ok = (1 in self.src_xfer_side_handles and self.src_xfer_side_handles[1] is not None)
                        dst_ok = (dst_engine_id in self.dst_xfer_side_handles and
                                  rr in self.dst_xfer_side_handles[dst_engine_id] and
                                  self.dst_xfer_side_handles[dst_engine_id][rr] is not None)
                        nb_ok = (dst_engine_id in self.dst_num_blocks)
                        if src_ok and dst_ok and nb_ok:
                            break
                        last_missing = f"down_ready(src={src_ok}, dst={dst_ok}, nb={nb_ok}, rr={rr})"
                    else:
                        tp_dst = self._tp_size.get(dst_engine_id)
                        tp_src = self._tp_size.get(self.engine_id)
                        if tp_dst is not None and tp_src is not None and tp_src > 0:
                            tp_mult = tp_dst // tp_src
                            eff_tp = max(1, tp_mult)
                            src_ok = (tp_mult in self.src_xfer_side_handles
                                      and self.src_xfer_side_handles[tp_mult] is not None)
                            dst_map = self.dst_xfer_side_handles.get(dst_engine_id) or {}
                            dst_ok = all((i in dst_map and dst_map[i] is not None) for i in range(eff_tp))
                            nb_ok = (dst_engine_id in self.dst_num_blocks)
                            if tp_mult >= 1 and src_ok and dst_ok and nb_ok:
                                break
                            last_missing = (f"up_ready(tp_dst={tp_dst}, tp_src={tp_src}, "
                                            f"tp_mult={tp_mult}, src={src_ok}, dst={dst_ok}, nb={nb_ok})")
                        else:
                            # —— 再尝试一次晚到的收养，避免竞态
                            try:
                                if self._adopt_remote_md_from_cache(dst_engine_id):
                                    logger.info("[WRITE][ADOPT] late-adopted metadata for dst=%s", dst_engine_id)
                                    continue
                            except Exception as _e:
                                logger.debug("[WRITE][ADOPT] late adopt failed for dst=%s: %s", dst_engine_id, _e)
                            last_missing = f"tp_size_missing(dst_has={tp_dst is not None}, src_has={tp_src is not None})"

                    if (time.time() - t0) * 1000.0 > wait_ms:
                        raise RuntimeError(
                            f"[WRITE] precondition not met on rank={self.rank} dst={dst_engine_id}: {last_missing} ; "
                            f"_tp_size_keys={list(self._tp_size.keys())} src_keys={list(self.src_xfer_side_handles.keys())} "
                            f"dst_keys_top={list(self.dst_xfer_side_handles.keys())}"
                        )
                    time.sleep(0.001)

                # ---- 真正传输 ----
                def _to_notify_str(x):
                    return x if isinstance(x, str) else str(x)

                if self._downscale_info.get(dst_engine_id) is not None:
                    info = self._downscale_info[dst_engine_id]
                    remote_rank = info["remote_rank"]
                    if 1 not in self.src_xfer_side_handles or self.src_xfer_side_handles[1] is None:
                        raise RuntimeError(f"[WRITE] DOWN missing src handle (rank={self.rank})")
                    if (dst_engine_id not in self.dst_xfer_side_handles or
                            remote_rank not in self.dst_xfer_side_handles[dst_engine_id]):
                        raise RuntimeError(f"[WRITE] DOWN missing dst handle (rank={self.rank} rr={remote_rank})")

                    self._write_blocks_down(local_block_ids, remote_block_ids, dst_engine_id, notify_msg)
                    if os.getenv("NIXL_DOWN_VERIFY", "0") == "1":  # 默认关闭验证，避免打断吞吐
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

                # ===== UP / EQ =====
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
                    logger.info("[WRITE] zero-block notify sent (tp=%s)", tp_multiplier)
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
                handles = []
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

                # 等待全部完成
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
                            raise RuntimeError(f"[WRITE] transfer failed state={st}")
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

    def _normalize_kv_rows(self, engine_id: str, rows, agent_tp: int):
        """
        允许两种输入：
          - 三维: [tp][layers][entries]  -> 直接返回
          - 二维: [layers][entries] 且 agent_tp==1 -> 自动包一层变成 [1][layers][entries]
        其它形状一律报错，防止静默错配。
        """
        if rows is None:
            return []

        # 三维且外层等于 agent_tp：认为已是规范形状
        if (isinstance(rows, list) and rows and isinstance(rows[0], list)
                and len(rows) == int(agent_tp) and isinstance(rows[0][0], (int,)) is False):
            return rows

        # 二维且 agent_tp==1：若维度与本地模型一致，自动包一层
        if (int(agent_tp) == 1 and isinstance(rows, list)
                and len(rows) == int(self.num_layers)
                and all(isinstance(x, list) and len(x) == int(self.num_cache_entries) for x in rows)):
            return [rows]

        raise RuntimeError(
            f"[ADD] kv_caches_base_addr shape incompatible with agent_tp={agent_tp}: "
            f"outer_len={len(rows) if isinstance(rows, list) else 'NA'} "
            f"(expect {agent_tp} or {self.num_layers} when tp==1)"
        )

    def add_remote_agent(
            self,
            engine_id: str,
            agent_metadata: List[bytes],
            agent_tp: int,
            kv_caches_base_addr: List[List[List[int]]],
            num_blocks: int,
            kv_caches_dev_ids: Optional[List[List[List[int]]]] = None,
    ):
        """
        幂等 + 可复用：
        - 不重复注册远端 agent / dlist，但**每个 rank** 都会补齐自己的 down/up 必要句柄与元数据。
        - 环境变量映射优先：NIXL_MAP_VLLMWORKER / NIXL_MAP_PREFILLWORKER；否则可从 kv_caches_dev_ids 或 pool_len 回退。
        """
        with self._timing.span("add_remote_agent"):
            agent_metadata = self._coerce_agent_metadata(agent_metadata)

            # 记录对端 TP（每次都写，幂等）
            self._tp_size[engine_id] = int(agent_tp)

            pid = os.getpid()
            try:
                dev = int(torch.cuda.current_device())
            except Exception:
                dev = None
            logger.info("[ADD][ENTER] pid=%s local_rank=%s cuda_dev=%s", pid, self.rank, dev)
            logger.info("[ADD] num_blocks=%d dev_ids=%s", num_blocks, "Y" if kv_caches_dev_ids is not None else "N")
            logger.info("[ADD] engine=%s local_rank=%s local_tp=%s agent_tp=%s is_mla=%s",
                        engine_id, self.rank, self._tp_size[self.engine_id], agent_tp, self._is_mla)

            # 严格 engine_id 复用检查（不同指纹报错/警告）
            self._check_engine_id_reuse(engine_id, agent_metadata, agent_tp)

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

            # 保存对端元数据（幂等）
            self.kv_caches_base_addr[engine_id] = kv_caches_base_addr
            self.kv_caches_dev_ids[engine_id] = kv_caches_dev_ids if kv_caches_dev_ids is not None else None
            loc_base = self.kv_caches_base_addr[engine_id]

            # 基本形状校验（确保 rank/L/E 维度正确）
            if len(agent_metadata) != agent_tp:
                raise RuntimeError(f"[ADD] agent_metadata len={len(agent_metadata)} != agent_tp={agent_tp}")
            if len(loc_base) != agent_tp:
                raise RuntimeError(f"[ADD] kv_caches_base_addr outer len={len(loc_base)} != agent_tp={agent_tp}")
            for r in range(agent_tp):
                assert len(loc_base[r]) == self.num_layers
                for L in range(self.num_layers):
                    assert len(loc_base[r][L]) == self.num_cache_entries

            # ========= 环境变量映射（仅用于 DOWN 的“远端 dev_id/pool-index”）=========
            def _parse_map(env_name: str) -> dict[int, int]:
                s = os.getenv(env_name, "").strip()
                if not s:
                    return {}
                out = {}
                for item in s.split(","):
                    kv = item.strip()
                    if not kv:
                        continue
                    if "->" in kv:
                        a, b = kv.split("->", 1)
                    elif ":" in kv:
                        a, b = kv.split(":", 1)
                    else:
                        logger.warning("[MAP] skip invalid pair %r in %s", kv, env_name)
                        continue
                    try:
                        out[int(a.strip())] = int(b.strip())
                    except Exception:
                        logger.warning("[MAP] skip invalid ints %r in %s", kv, env_name)
                return out

            def _pool_len_for_role(role: str) -> int:
                names = ["NIXL_POOL_VLLMWORKER", "NIXL_POOL_PREFILLWORKER", "NIXL_POOL"]
                s = next((os.getenv(n) for n in names if os.getenv(n)), None)
                if not s:
                    return 0
                try:
                    arr = [x.strip() for x in s.split(",") if x.strip()]
                    return len(arr)
                except Exception:
                    return 0

            remote_role = "VLLMWORKER" if int(agent_tp) == 1 else "PREFILLWORKER"
            ENV_MAP_NAME = "NIXL_MAP_VLLMWORKER" if remote_role == "VLLMWORKER" else "NIXL_MAP_PREFILLWORKER"
            _env_map = _parse_map(ENV_MAP_NAME)
            _pool_len_hint = _pool_len_for_role(remote_role)

            def _remote_pool_index_by_env_or_md(r_engine_id: str, r_idx: int, layer: int, entry_idx: int) -> int:
                # 1) kv_caches_dev_ids 优先（若提供）
                devs = self.kv_caches_dev_ids.get(r_engine_id)
                if devs is not None:
                    try:
                        v = int(devs[r_idx][layer][entry_idx])
                        return v
                    except Exception:
                        logger.warning("[ADD] invalid kv_caches_dev_ids for engine=%s r=%d L=%d E=%d",
                                       r_engine_id, r_idx, layer, entry_idx)
                # 2) 显式环境映射
                if r_idx in _env_map:
                    v = int(_env_map[r_idx])
                    if _pool_len_hint and v >= _pool_len_hint:
                        logger.warning("[MAP] %s maps %d->%d out of pool_len=%d",
                                       ENV_MAP_NAME, r_idx, v, _pool_len_hint)
                    return v
                # 3) 回退：用 worker_idx 取模（可预测）
                if _pool_len_hint:
                    v = r_idx % _pool_len_hint
                    logger.info("[MAP][FALLBACK] %s not set for %d, fallback pool_index=%d (pool_len=%d)",
                                ENV_MAP_NAME, r_idx, v, _pool_len_hint)
                    return v
                # 4) 无法得知 pool 长度时，兜底 0
                logger.info("[MAP][FALLBACK] %s empty and pool_len unknown; use 0", ENV_MAP_NAME)
                return 0

            # ===== 注册远端 agent（重型步骤只做一次；但本 rank 的句柄/状态每次幂等补齐）=====
            if not self._engine_prepped.get(engine_id, False):
                agent_names: List[str] = []
                for meta in agent_metadata:
                    agent_names.append(self.nixl_wrapper.add_remote_agent(meta))
                self._remote_agents[engine_id] = agent_names
                logger.info("[ADD] remote_agents registered: dst_engine=%s count=%d names_sample=%s",
                            engine_id, len(agent_names), self._peek(agent_names, 3))
                self._engine_prepped[engine_id] = True
            else:
                logger.info("[ADD][IDEMPOTENT] reuse remote_agents engine=%s", engine_id)

            # ===== 路径分岔 =====
            tp_multiplier = self._tp_size[engine_id] // self._tp_size[self.engine_id]
            logger.info("[ADD] tp_multiplier=%s (dst_tp/src_tp = %s/%s)",
                        tp_multiplier, self._tp_size[engine_id], self._tp_size[self.engine_id])

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

                # 每次都把 downscale_info 补齐（幂等）
                self._downscale_info[engine_id] = {
                    "group_size": group_size,
                    "remote_rank": remote_rank,
                    "peer_idx": peer_idx,
                    "notify_leader": (peer_idx == 0),
                    "perm": None,
                    "token_granularity": True,
                }

                # 目的（decode）的 token 粒度单位数（幂等）
                self.dst_num_blocks[engine_id] = num_blocks * B
                logger.info(
                    "[ADD][DOWN] group_size=%d remote_rank=%d peer_idx=%d token_len_local=%d token_len_total=%d full_len=%d seg_len=%d peer_off_tok=%d",
                    group_size, remote_rank, peer_idx, token_len_local, token_len_total, full_len, seg_len,
                    peer_off_tok)

                # src dlist（只建一次）—— 本地 dev_id 用 self.rank
                if 1 not in self.src_xfer_side_handles or self.src_xfer_side_handles[1] is None:
                    src_blocks = []
                    local_dev_id = self.rank
                    for layer in range(self.num_layers):
                        # 本地是标准布局：每 layer 有 [K, V] 两段基址
                        for base in self.kv_caches_base_addr[self.engine_id][layer]:  # K、V
                            for bid in range(self.num_blocks):
                                base_block = base + bid * seg_len
                                for t in range(B):
                                    src_blocks.append((base_block + t * token_len_local, token_len_local, local_dev_id))
                    desc = self.nixl_wrapper.get_xfer_descs(src_blocks, "VRAM")
                    self.src_xfer_side_handles[1] = self.nixl_wrapper.prep_xfer_dlist("", desc)

                # dst dlist（只建一次）—— 远端 dev_id 用环境变量映射（或回退）
                if engine_id not in self.dst_xfer_side_handles:
                    self.dst_xfer_side_handles[engine_id] = {}
                if remote_rank not in self.dst_xfer_side_handles[engine_id]:
                    dst_blocks = []
                    for layer in range(self.num_layers):
                        layer_bases = self.kv_caches_base_addr[engine_id][remote_rank][layer]
                        for entry_idx, rbase in enumerate(layer_bases):  # K、V
                            r_pool_idx = _remote_pool_index_by_env_or_md(engine_id, remote_rank, layer, entry_idx)
                            for bid in range(num_blocks):
                                base_block = rbase + bid * full_len
                                for t in range(B):
                                    dst_blocks.append((base_block + t * token_len_total + peer_off_tok,
                                                       token_len_local, r_pool_idx))
                    desc = self.nixl_wrapper.get_xfer_descs(dst_blocks, "VRAM")
                    self.dst_xfer_side_handles[engine_id][remote_rank] = self.nixl_wrapper.prep_xfer_dlist(
                        self._remote_agents[engine_id][remote_rank], desc
                    )
                    try:
                        self.nixl_wrapper.make_connection(self._remote_agents[engine_id][remote_rank])
                    except Exception as e:
                        logger.debug("make_connection lazy: %s", e)

                # READ-DOWN 本地目的 dlist（token 颗粒）—— 本地 dev_id 用 self.rank
                if "read_down_src" not in self.src_xfer_side_handles or self.src_xfer_side_handles[
                    "read_down_src"] is None:
                    B = int(self.block_size)
                    H_loc = int(self.num_heads)  # 本机 shard 的头数
                    C = int(self.head_dim)
                    e = self.kv_caches[0][0].element_size()  # dtype 字节
                    token_len_local = H_loc * C * e  # 每 token 本机应占的连续字节
                    seg_len_local = B * token_len_local  # 本机一个 block 的总字节（按 token 拼起来）

                    blocks_local = []
                    local_dev_id = self.rank
                    for layer in range(self.num_layers):
                        for base in self.kv_caches_base_addr[self.engine_id][layer]:  # K / V 两段
                            for bid in range(self.num_blocks):
                                base_block = base + bid * seg_len_local
                                for t in range(B):
                                    blocks_local.append(
                                        (base_block + t * token_len_local, token_len_local, local_dev_id))
                    descs_local = self.nixl_wrapper.get_xfer_descs(blocks_local, "VRAM")
                    self.src_xfer_side_handles["read_down_src"] = self.nixl_wrapper.prep_xfer_dlist("", descs_local)

                # READ-DOWN 远端来源 dlist（token 颗粒）—— 远端 dev_id 用映射（或回退）
                if "read_down_dst" not in self.dst_xfer_side_handles[engine_id]:
                    B = int(self.block_size)
                    H_loc = int(self.num_heads)
                    C = int(self.head_dim)
                    e = self.kv_caches[0][0].element_size()
                    token_len_local = H_loc * C * e
                    token_len_total = group_size * token_len_local  # 远端 decode 的每 token 字节（全头）
                    seg_len_total = B * token_len_total
                    peer_off = peer_idx * token_len_local  # 本 peer 在“每 token”里的头切片偏移

                    blocks_remote = []
                    for layer in range(self.num_layers):
                        layer_bases = self.kv_caches_base_addr[engine_id][remote_rank][layer]  # [K_base, V_base]
                        for entry_idx, rbase in enumerate(layer_bases):
                            r_pool_idx = _remote_pool_index_by_env_or_md(engine_id, remote_rank, layer, entry_idx)
                            for bid in range(num_blocks):
                                base_block = rbase + bid * seg_len_total
                                for t in range(B):
                                    blocks_remote.append((base_block + t * token_len_total + peer_off,
                                                          token_len_local, r_pool_idx))
                    descs_remote = self.nixl_wrapper.get_xfer_descs(blocks_remote, "VRAM")
                    self.dst_xfer_side_handles[engine_id]["read_down_dst"] = self.nixl_wrapper.prep_xfer_dlist(
                        self._remote_agents[engine_id][remote_rank], descs_remote
                    )
                    try:
                        self.nixl_wrapper.make_connection(self._remote_agents[engine_id][remote_rank])
                    except Exception as e:
                        logger.debug("[ADD][READ-DOWN] make_connection lazy: %s", e)

                # 记录远端的“块数”（索引仍按 block_ids 进来，展开到 token 索引在 READ 阶段做）
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
                return self._remote_agents[engine_id]

            # --------- UP / EQ（tp_multiplier > 0）---------
            assert tp_multiplier > 0, f"[ADD] invalid tp_multiplier={tp_multiplier}"
            dst_block_len = self.block_len if self._is_mla else (self.block_len // tp_multiplier)
            logger.info("[ADD] up/equal path: dst_block_len=%s", dst_block_len)

            # src 侧 dlist（按 tp_multiplier 切分），只建一次
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

            # 远端 dlist（每个 i 只建一次）
            self.dst_num_blocks[engine_id] = num_blocks
            if engine_id not in self.dst_xfer_side_handles:
                self.dst_xfer_side_handles[engine_id] = {}
            for i in range(tp_multiplier):
                if i in self.dst_xfer_side_handles[engine_id] and self.dst_xfer_side_handles[engine_id][i] is not None:
                    continue
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
            return self._remote_agents[engine_id]

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
