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

from .kv_rearrange import rearrange_tensors, _grouped_to_standard, _rearrange_kernel_write_down
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

    # --------- NEW: retry wrapper for prep_xfer_dlist (fix NOT_FOUND races) ---------
    def _prep_dlist_retry(self, agent_name, descs, backends=None,
                         tries: int = 80, sleep_s: float = 0.02, sleep_max: float = 0.2):
        """
        prep_xfer_dlist 在对端 register_memory 尚未 ready 时可能短暂 NIXL_ERR_NOT_FOUND。
        这里做指数退避重试，避免 add_remote_agent 竞态直接炸。
        """
        last = None
        for _ in range(max(1, int(tries))):
            try:
                if backends is None:
                    return self.nixl_wrapper.prep_xfer_dlist(agent_name, descs)
                return self.nixl_wrapper.prep_xfer_dlist(agent_name, descs, backends=backends)
            except Exception as e:
                last = e
                if "NIXL_ERR_NOT_FOUND" in str(e):
                    time.sleep(sleep_s)
                    sleep_s = min(sleep_s * 1.4, sleep_max)
                    continue
                raise
        raise last

    # --------- NEW: choose remote dev_id for VRAM desc ---------
    def _remote_devid(self, remote_rank: int, pool_idx: int) -> int:
        """
        默认使用 remote_rank 做 dev_id（最符合 vLLM/NIXL 常规注册方式）。
        如确实需要用 pool/mapping（非常规环境），设置：
          NIXL_REMOTE_DEVID_MODE=pool
        """
        mode = os.getenv("NIXL_REMOTE_DEVID_MODE", "rank").strip().lower()
        if mode in ("pool", "map", "mapping"):
            return int(pool_idx)
        return int(remote_rank)

    def _wait_many(self, handles):
        with self._timing.span("wait_many"):
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

    def _ensure_remote_md_ready(self, dst_engine_id: str) -> None:
        """
        在 read/write 使用 tp 之前确保已有 dst 的 tp_size。
        先看内存缓存，再尝试从共享缓存收养（调用你已有的 _adopt_remote_md_from_cache）。
        """
        # 本地 tp 也顺便兜底一下，避免后面除法出 KeyError
        if self.engine_id not in self._tp_size and hasattr(self, "_local_tp_size"):
            self._tp_size[self.engine_id] = int(self._local_tp_size)

        # 1) 已有就直接返回
        tp = self._tp_size.get(dst_engine_id)
        if isinstance(tp, int) and tp > 0:
            return

        # 2) engine_meta 里有，取一遍
        meta = getattr(self, "_engine_meta", {}).get(dst_engine_id)
        if isinstance(meta, dict):
            cand = meta.get("agent_tp") or meta.get("tp_size") or meta.get("tp")
            if cand is not None:
                try:
                    self._tp_size[dst_engine_id] = int(cand)
                    return
                except Exception:
                    pass

        # 3) 尝试从共享缓存收养（你已经实现好的）
        if hasattr(self, "_adopt_remote_md_from_cache"):
            adopted = self._adopt_remote_md_from_cache(dst_engine_id)
            if adopted:
                tp2 = self._tp_size.get(dst_engine_id)
                if isinstance(tp2, int) and tp2 > 0:
                    return
                # 退一步从 engine_meta 再拿一次
                meta2 = getattr(self, "_engine_meta", {}).get(dst_engine_id)
                if isinstance(meta2, dict):
                    cand2 = meta2.get("agent_tp") or meta2.get("tp_size") or meta2.get("tp")
                    if cand2 is not None:
                        try:
                            self._tp_size[dst_engine_id] = int(cand2)
                            return
                        except Exception:
                            pass

        # 4) 仍然没有就报出清晰错误
        known = list(self._tp_size.keys())
        raise RuntimeError(
            f"[READ] missing tp_size for dst_engine={dst_engine_id}; "
            f"known_tp_entries={known}. "
            f"ensure metadata file exists and adoptable before first read."
        )

    def _ensure_down_ready(self, dst_engine_id: str) -> None:
        """确保 DOWN 路的本地/远端句柄与元数据就绪；否则尝试从共享缓存收养并复检。"""

        def _have_all() -> bool:
            down = self._downscale_info.get(dst_engine_id)
            if down is None:
                return False
            rr = down.get("remote_rank")
            # 写入用的 token 句柄
            if 1 not in self.src_xfer_side_handles or self.src_xfer_side_handles[1] is None:
                return False
            if (dst_engine_id not in self.dst_xfer_side_handles or
                    rr not in self.dst_xfer_side_handles[dst_engine_id] or
                    self.dst_xfer_side_handles[dst_engine_id][rr] is None):
                return False
            if dst_engine_id not in self.dst_num_blocks:
                return False
            # 校验/READ-DOWN 需要的句柄
            if ("read_down_src" not in self.src_xfer_side_handles or
                    self.src_xfer_side_handles["read_down_src"] is None):
                return False
            if ("read_down_dst" not in self.dst_xfer_side_handles.get(dst_engine_id, {}) or
                    self.dst_xfer_side_handles[dst_engine_id]["read_down_dst"] is None):
                return False
            return True

        with self._timing.span("write_down.ensure_ready"):
            if not _have_all():
                try:
                    # 尝试从 /dev/shm|/tmp 的 msgpack 缓存“收养”，内部会调用 add_remote_agent(DOWN)
                    self._adopt_remote_md_from_cache(dst_engine_id)
                except Exception as e:
                    logger.debug("[DOWN-ENSURE] adopt failed for %s: %s", dst_engine_id, e)
            if not _have_all():
                down = self._downscale_info.get(dst_engine_id)
                rr = (down or {}).get("remote_rank")
                raise RuntimeError(
                    f"[DOWN-READY] not ready for dst={dst_engine_id}: "
                    f"down={'Y' if down else 'N'} "
                    f"src_tok={'Y' if 1 in self.src_xfer_side_handles else 'N'} "
                    f"src_read={'Y' if 'read_down_src' in self.src_xfer_side_handles else 'N'} "
                    f"dst_tok={'Y' if (dst_engine_id in self.dst_xfer_side_handles and rr in self.dst_xfer_side_handles[dst_engine_id]) else 'N'} "
                    f"dst_read={'Y' if ('read_down_dst' in self.dst_xfer_side_handles.get(dst_engine_id, {})) else 'N'} "
                    f"dst_nb={'Y' if dst_engine_id in self.dst_num_blocks else 'N'}"
                )

    # -------------------- PATCH 1/2: 修复 all_gather group 失败时误用 WORLD --------------------
    def _write_blocks_down(self, local_block_ids, remote_block_ids, dst_engine_id, notify_msg):
        """
        Downscale (prefill -> decode) 写入：
        - group 内 all-gather 聚合，只有 leader 单写 decode
        - NEW: 如果子 group 创建失败/pg 不可用，**不允许 fallback 到 WORLD all_gather**，而是退回 token-fallback
        """
        import torch.distributed as dist

        with self._timing.span("write_down"):
            self._ensure_down_ready(dst_engine_id)
            info = self._downscale_info[dst_engine_id]
            assert info is not None, "[WRITE-DOWN] downscale info missing"

            if len(local_block_ids) == 0:
                notify_payload = notify_msg if isinstance(notify_msg, (str, bytes)) else str(notify_msg)
                if info.get("notify_leader") and notify_payload:
                    ra = self._remote_agents[dst_engine_id][info["remote_rank"]]
                    self.nixl_wrapper.send_notif(
                        ra,
                        notify_payload if isinstance(notify_payload, bytes) else notify_payload.encode("utf-8")
                    )
                return

            assert len(local_block_ids) == len(remote_block_ids), \
                f"[WRITE-DOWN] token len mismatch: local={len(local_block_ids)} remote={len(remote_block_ids)}"

            remote_rank = int(info["remote_rank"])
            group_size = int(info["group_size"])
            peer_idx = int(info["peer_idx"])
            is_leader = bool(info.get("notify_leader"))

            BACKENDS = ["UCX"] if os.getenv("NIXL_FORCE_UCX", "1") == "1" else []

            B = int(self.block_size)
            H_loc = int(self.num_heads)
            C = int(self.head_dim)
            ebytes = int(self.kv_caches[0][0].element_size())
            H_total = H_loc * max(1, group_size)

            token_len_local = H_loc * C * ebytes
            token_len_total = H_total * C * ebytes
            seg_len_total = B * token_len_total

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
                        continue
                    try:
                        out[int(a.strip())] = int(b.strip())
                    except Exception:
                        continue
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

            def _remote_pool_index_by_env_or_md(r_engine_id: str, r_idx: int, layer: int, entry_idx: int) -> int:
                devs = self.kv_caches_dev_ids.get(r_engine_id)
                if devs is not None:
                    try:
                        return int(devs[r_idx][layer][entry_idx])
                    except Exception:
                        pass
                agent_tp = int(self._tp_size.get(r_engine_id, 1))
                remote_role = "VLLMWORKER" if agent_tp == 1 else "PREFILLWORKER"
                ENV_MAP_NAME = "NIXL_MAP_VLLMWORKER" if remote_role == "VLLMWORKER" else "NIXL_MAP_PREFILLWORKER"
                _env_map = _parse_map(ENV_MAP_NAME)
                _pool_len_hint = _pool_len_for_role(remote_role)
                if r_idx in _env_map:
                    return int(_env_map[r_idx])
                if _pool_len_hint:
                    return r_idx % _pool_len_hint
                return 0

            notify_payload = notify_msg if isinstance(notify_msg, (str, bytes)) else str(notify_msg)
            with self._timing.span("write_down.barrier"):
                self._barrier_mark_and_wait(
                    dst_engine_id,
                    (notify_payload if isinstance(notify_payload, str) else str(notify_payload)),
                    group_size,
                    peer_idx,
                    is_leader
                )

            # ---- decide fast path ----
            use_fast_path = (group_size > 1) and hasattr(dist, "is_available") and dist.is_available() and dist.is_initialized()

            # ---- try create pg; if fail, DO NOT fall back to WORLD gather, but token-fallback ----
            pg = None
            if use_fast_path:
                base_rank = (self.rank // group_size) * group_size
                ranks_group = list(range(base_rank, base_rank + group_size))
                pg = info.get("pg")
                if pg is None:
                    try:
                        pg = dist.new_group(ranks=ranks_group)
                        info["pg"] = pg
                    except Exception as e:
                        logger.warning("[DOWN] new_group failed (%s), fallback to token path (NOT world)", e)
                        pg = None
                        use_fast_path = False
                if pg is None:
                    logger.warning("[DOWN] pg is None, fallback to token path (NOT world)")
                    use_fast_path = False

            # ---- token fallback (原逻辑整体保留) ----
            if not use_fast_path:
                with self._timing.span("write_down.expand_tokens"):
                    token_ids_local = self._expand_blocks_to_tokens_cached(local_block_ids)
                    token_ids_remote = self._expand_blocks_to_tokens_cached(remote_block_ids)
                Ntok = len(token_ids_local)
                if Ntok == 0:
                    if is_leader and notify_payload:
                        ra = self._remote_agents[dst_engine_id][remote_rank]
                        self.nixl_wrapper.send_notif(
                            ra,
                            notify_payload if isinstance(notify_payload, bytes) else notify_payload.encode("utf-8")
                        )
                    return

                if 1 not in self.src_xfer_side_handles or self.src_xfer_side_handles[1] is None:
                    raise RuntimeError(f"[WRITE-DOWN] missing src token handle (rank={self.rank})")
                if (dst_engine_id not in self.dst_xfer_side_handles or
                        remote_rank not in self.dst_xfer_side_handles[dst_engine_id] or
                        self.dst_xfer_side_handles[dst_engine_id][remote_rank] is None):
                    raise RuntimeError(f"[WRITE-DOWN] missing dst token handle (rank={self.rank} rr={remote_rank})")
                src_hdl = self.src_xfer_side_handles[1]
                dst_hdl = self.dst_xfer_side_handles[dst_engine_id][remote_rank]

                per_entry_src = int(self.num_blocks) * int(self.block_size)
                per_entry_dst = int(self.dst_num_blocks[dst_engine_id])
                same_layout = (per_entry_src == per_entry_dst and local_block_ids == remote_block_ids)

                with self._timing.span("write_down.build_idx"):
                    le_list = self._get_le_list()
                    local_idx: list[int] = []
                    remote_idx: list[int] | None = None if same_layout else []
                    for (layer, entry) in le_list:
                        base_layer_src = layer * (self.num_cache_entries * per_entry_src)
                        base_layer_dst = layer * (self.num_cache_entries * per_entry_dst)
                        base_entry_src = base_layer_src + entry * per_entry_src
                        base_entry_dst = base_layer_dst + entry * per_entry_dst
                        for t in token_ids_local:
                            local_idx.append(base_entry_src + t)
                        if remote_idx is not None:
                            for t in token_ids_remote:
                                remote_idx.append(base_entry_dst + t)

                with self._timing.span("write_down.submit"):
                    h = self.nixl_wrapper.make_prepped_xfer(
                        "WRITE",
                        src_hdl, local_idx,
                        dst_hdl, (local_idx if same_layout else remote_idx),
                        (notify_payload if is_leader else b"") if isinstance(notify_payload, bytes)
                        else ((notify_payload.encode("utf-8") if is_leader else b"")),
                        backends=BACKENDS
                    )
                    self.nixl_wrapper.transfer(h)
                with self._timing.span("write_down.wait_window"):
                    self._wait_many([h])
                logger.info("[WRITE][DOWN-single-token] leader=%s Ntok=%d LxE=%d", is_leader,
                            Ntok, self.num_layers * self.num_cache_entries)
                return

            # ---- fast path: pg 一定可用；不允许无 group 的 WORLD gather ----
            base_rank = (self.rank // group_size) * group_size
            ranks_group = list(range(base_rank, base_rank + group_size))
            ra_decode = self._remote_agents[dst_engine_id][remote_rank]

            CHUNK_BLKS = int(os.getenv("NIXL_DOWN_PACK_CHUNK_BLOCKS", "16"))
            if CHUNK_BLKS <= 0:
                CHUNK_BLKS = len(local_block_ids)

            for L in range(self.num_layers):
                for entry_idx in range(self.num_cache_entries):
                    rbase = int(self.kv_caches_base_addr[dst_engine_id][remote_rank][L][entry_idx])

                    pool_idx = _remote_pool_index_by_env_or_md(dst_engine_id, remote_rank, L, entry_idx)
                    rdev = self._remote_devid(remote_rank, pool_idx)  # default rank

                    off = 0
                    while off < len(local_block_ids):
                        hi = min(off + CHUNK_BLKS, len(local_block_ids))
                        lb = local_block_ids[off:hi]
                        rb = remote_block_ids[off:hi]
                        nblk = len(lb)

                        with self._timing.span("write_down.pack_local"):
                            src = self.kv_caches[L][entry_idx][lb].contiguous()
                            pack = src.permute(0, 2, 1, 3).contiguous().reshape(nblk, H_loc, B * C)

                        with self._timing.span("write_down.gather"):
                            recv_list = [torch.empty_like(pack) for _ in range(group_size)]
                            # ✅ 只用 pg；绝不 WORLD
                            dist.all_gather(recv_list, pack, group=pg)

                        if is_leader:
                            gathered = torch.cat(recv_list, dim=1)
                            std = gathered.reshape(nblk, H_total, B, C).permute(0, 2, 1, 3).contiguous()

                            reg_h = None
                            try:
                                reg_h = self.nixl_wrapper.register_memory([std])
                                src_slices = [std[j] for j in range(nblk)]
                                src_desc = self.nixl_wrapper.get_xfer_descs(list(src_slices))
                                src_h = self.nixl_wrapper.prep_xfer_dlist("", src_desc, backends=BACKENDS)

                                dst_tuples = []
                                for j, b_id in enumerate(rb):
                                    dst_addr = rbase + int(b_id) * seg_len_total
                                    dst_len = seg_len_total
                                    dst_tuples.append((dst_addr, dst_len, int(rdev)))
                                dst_desc = self.nixl_wrapper.get_xfer_descs(dst_tuples, "VRAM")
                                dst_h = self._prep_dlist_retry(ra_decode, dst_desc, backends=BACKENDS)

                                last_chunk = (hi >= len(local_block_ids))
                                last_layer = (L == self.num_layers - 1)
                                last_entry = (entry_idx == self.num_cache_entries - 1)
                                piggy = (
                                    notify_msg if isinstance(notify_msg, bytes) else str(notify_msg).encode("utf-8")
                                ) if (is_leader and last_chunk and last_layer and last_entry) else b""

                                idx = list(range(nblk))
                                with self._timing.span("write_down.submit"):
                                    h = self.nixl_wrapper.make_prepped_xfer(
                                        "WRITE",
                                        src_h, idx,
                                        dst_h, idx,
                                        piggy,
                                        backends=BACKENDS,
                                        skip_desc_merge=True,
                                    )
                                    self.nixl_wrapper.transfer(h)
                                with self._timing.span("write_down.wait_window"):
                                    self._wait_many([h])
                            finally:
                                if reg_h is not None:
                                    try:
                                        self.nixl_wrapper.deregister_memory(reg_h)
                                    except Exception as _e:
                                        logger.debug("[DOWN] deregister std-chunk failed: %s", _e)

                        off = hi

            if is_leader:
                logger.info(
                    "[WRITE][DOWN-fast] leader rank=%d remote_rank=%d group=%s -> single-write per block/entry/layer",
                    self.rank, remote_rank, ranks_group)
            else:
                logger.info("[WRITE][DOWN-fast] follower rank=%d done (no decode write)", self.rank)

    # -------------------- read down / others unchanged --------------------
    def _read_blocks_down(self, local_block_ids, staging_block_ids, remote_block_ids, dst_engine_id):
        """
        Prefill<--READ--Decode（Downscale）路径：
        - 采用 UCX 后端（可通过 NIXL_FORCE_UCX=0 关闭）
        - 大 dlist 分块 + 有限并发窗口
        - 读完在本机做 standard->grouped 的 GPU 重排
        """
        with self._timing.span("read_down"):
            self._ensure_down_ready(dst_engine_id)
            down = self._downscale_info.get(dst_engine_id)
            assert down is not None, "[READ-DOWN] downscale info missing"

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

            dst_handle = self.src_xfer_side_handles["read_down_src"]
            src_handle = self.dst_xfer_side_handles[dst_engine_id]["read_down_dst"]

            def _ids_blockwise(num_blocks_total: int, block_ids: list[int]) -> list[int]:
                ids = []
                for layer in range(self.num_layers):
                    for entry in range(self.num_cache_entries):
                        for b in block_ids:
                            ids.append(layer * self.num_cache_entries * num_blocks_total
                                       + entry * num_blocks_total + int(b))
                return ids

            num_blocks_remote = int(self.dst_num_blocks_read[dst_engine_id])
            src_desc_ids = _ids_blockwise(num_blocks_remote, remote_block_ids)
            dst_desc_ids = _ids_blockwise(self.num_blocks, staging_block_ids)
            if len(src_desc_ids) != len(dst_desc_ids):
                raise RuntimeError(f"[READ-DOWN] desc len mismatch: src={len(src_desc_ids)} dst={len(dst_desc_ids)}")

            MAX_IOV_RD = int(os.getenv("NIXL_MAX_IOV_READ", "16384"))
            MAX_INFLIGHT_RD = int(os.getenv("NIXL_MAX_INFLIGHT_READ", "8"))
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

            ngroups = int(down.get("group_size", 1))
            if ngroups <= 1 or self._is_mla:
                return

            try:
                sample = self.kv_caches[0][0]
                H = int(sample.shape[2])
            except Exception:
                H = None
            if not H or (H % ngroups != 0):
                logger.warning("[READ-DOWN] skip rearrange: invalid H=%s for ngroups=%s (must divide).", H, ngroups)
                return

            local_ranges = self._get_ranges(local_block_ids)
            staging_ranges = self._get_ranges(staging_block_ids)
            if len(local_ranges) != len(staging_ranges):
                n = min(len(local_ranges), len(staging_ranges))
                local_ranges, staging_ranges = local_ranges[:n], staging_ranges[:n]

            CHUNK_BLKS = int(os.getenv("NIXL_READ_REARRANGE_CHUNK_BLOCKS", "16"))
            from .kv_rearrange import rearrange_tensors_read_down

            for (l0, l1), (s0, s1) in zip(local_ranges, staging_ranges):
                n_blocks = (l1 - l0 + 1)
                if CHUNK_BLKS > 0:
                    steps = range(l0, l1 + 1, CHUNK_BLKS)
                else:
                    steps = [l0]

                for start in steps:
                    end = min(start + (CHUNK_BLKS if CHUNK_BLKS > 0 else n_blocks), l1 + 1)
                    s_start = s0 + (start - l0)
                    s_end = s_start + (end - start)

                    for kv_cache in self.kv_caches:
                        for cache in kv_cache:
                            t_std = cache[s_start:s_end].contiguous()
                            t_grp = cache[start:end].contiguous()
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

    # (略：中间函数保持你原样)
    # --------------------------- 省略到 _md_cache_dir/_persist_remote_md_cache 等 ---------------------------

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

    def _md_cache_dir(self) -> str:
        d = os.getenv("NIXL_MD_CACHE_DIR")
        if not d:
            d = "/dev/shm" if os.path.isdir("/dev/shm") else "/tmp"
        os.makedirs(d, exist_ok=True)
        return d

    def _md_cache_path(self, engine_id: str) -> str:
        safe_engine = self._sanitize_key(engine_id, 32)
        return os.path.join(self._md_cache_dir(), f"nixl_md_{safe_engine}.msgpack")

    # -------------------- PATCH 2/2: self-md 用 part 聚合成 3D，避免 adopt 失败 --------------------
    def _md_cache_part_path(self, engine_id: str, part_rank: int) -> str:
        safe_engine = self._sanitize_key(engine_id, 32)
        return os.path.join(self._md_cache_dir(), f"nixl_md_{safe_engine}.part{int(part_rank)}.msgpack")

    def _persist_md_part(
        self,
        engine_id: str,
        agent_tp: int,
        part_rank: int,
        agent_metadata_part: bytes,
        kv_caches_base_addr_part,
        num_blocks: int,
        kv_caches_dev_ids_part,
    ) -> None:
        payload = {
            "engine_id": engine_id,
            "agent_tp": int(agent_tp),
            "part_rank": int(part_rank),
            "agent_metadata_part": bytes(agent_metadata_part),
            "kv_caches_base_addr_part": kv_caches_base_addr_part,
            "num_blocks": int(num_blocks),
            "kv_caches_dev_ids_part": kv_caches_dev_ids_part,
        }
        b = msgspec.msgpack.encode(payload)
        path = self._md_cache_part_path(engine_id, part_rank)
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "wb") as f:
            f.write(b)
        os.replace(tmp, path)

    def _try_assemble_md_from_parts(self, engine_id: str, agent_tp: int):
        metas: List[bytes] = []
        rows3 = []
        dev_ids3 = []
        num_blocks = None

        for r in range(int(agent_tp)):
            p = self._md_cache_part_path(engine_id, r)
            if not os.path.exists(p):
                return None
            with open(p, "rb") as f:
                d = msgspec.msgpack.decode(f.read())
            if d.get("engine_id") != engine_id:
                return None
            if int(d.get("agent_tp", -1)) != int(agent_tp):
                return None
            if int(d.get("part_rank", -1)) != int(r):
                return None
            meta = d.get("agent_metadata_part", b"")
            metas.append(bytes(meta))
            rows3.append(d.get("kv_caches_base_addr_part"))
            dev_ids3.append(d.get("kv_caches_dev_ids_part"))
            if num_blocks is None:
                num_blocks = int(d.get("num_blocks", 0))

        # dev_ids3 可能是 None/形状不齐，这里简单取 None（你当前逻辑主要用 env-map/remote_rank）
        # 如你后续要严格用 dev_ids，可把它也做成 [tp][layers][entries] 再下发
        return {
            "engine_id": engine_id,
            "agent_tp": int(agent_tp),
            "agent_metadata": metas,
            "kv_caches_base_addr": rows3,
            "num_blocks": int(num_blocks or 0),
            "kv_caches_dev_ids": None,
        }

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
        os.replace(tmp, path)
        logger.debug("[MD-CACHE] persisted for engine=%s path=%s size=%dB", engine_id, path, len(b))

    def _coerce_agent_metadata(self, md) -> List[bytes]:
        """把各种可能的形式统一成 List[bytes]，避免把 bytes 当成可迭代的 int。"""
        if md is None:
            return []
        if isinstance(md, (bytes, bytearray, memoryview)):
            return [bytes(md)]
        if isinstance(md, (list, tuple)):
            out = []
            for x in md:
                if isinstance(x, (bytes, bytearray, memoryview)):
                    out.append(bytes(x))
                else:
                    if isinstance(x, list) and all(isinstance(i, int) for i in x):
                        out.append(bytes(x))
                    else:
                        raise TypeError(f"agent_metadata elem must be bytes-like, got {type(x).__name__}")
            return out
        raise TypeError(f"agent_metadata must be bytes or list of bytes, got {type(md).__name__}")

    def _publish_self_md_cache(self) -> None:
        """
        self publish：每个 tp rank 写 part 文件，然后尽力拼成 3D canonical md。
        解决你遇到的：agent_tp>1 但 kv_caches_base_addr 只有 2D（len==num_layers）导致 adopt 失败。
        """
        tp = int(self._tp_size[self.engine_id])
        # 这里默认 rank 就是 tp-rank（你现在的 vLLM 多数如此）
        part_rank = int(self.rank)
        md_list = self._coerce_agent_metadata(self.get_agent_metadata())
        meta_part = md_list[0] if md_list else b""
        rows_part = self.kv_caches_base_addr[self.engine_id]

        try:
            self._persist_md_part(
                engine_id=self.engine_id,
                agent_tp=tp,
                part_rank=part_rank,
                agent_metadata_part=meta_part,
                kv_caches_base_addr_part=rows_part,
                num_blocks=int(self.num_blocks),
                kv_caches_dev_ids_part=self.kv_caches_dev_ids.get(self.engine_id),
            )
        except Exception as e:
            logger.debug("[MD-CACHE][SELF] part persist failed: %s", e)
            return

        # tp==1：直接写 canonical（用 3D 形式也没问题）
        if tp <= 1:
            try:
                self._persist_remote_md_cache(
                    engine_id=self.engine_id,
                    agent_metadata=[meta_part],
                    kv_caches_base_addr=[rows_part],
                    num_blocks=int(self.num_blocks),
                    kv_caches_dev_ids=self.kv_caches_dev_ids.get(self.engine_id),
                    agent_tp=tp,
                )
                logger.info("[MD-CACHE][SELF] published engine=%s tp=%d (direct)", self.engine_id, tp)
            except Exception as e:
                logger.debug("[MD-CACHE][SELF] publish direct failed: %s", e)
            return

        # tp>1：尝试 assemble
        assembled = None
        try:
            assembled = self._try_assemble_md_from_parts(self.engine_id, tp)
        except Exception as e:
            logger.debug("[MD-CACHE][SELF] assemble try failed: %s", e)

        if assembled:
            try:
                self._persist_remote_md_cache(
                    engine_id=self.engine_id,
                    agent_metadata=assembled["agent_metadata"],
                    kv_caches_base_addr=assembled["kv_caches_base_addr"],
                    num_blocks=int(assembled["num_blocks"]),
                    kv_caches_dev_ids=assembled.get("kv_caches_dev_ids"),
                    agent_tp=int(assembled["agent_tp"]),
                )
                logger.info("[MD-CACHE][SELF] published engine=%s tp=%d (assembled)", self.engine_id, tp)
            except Exception as e:
                logger.debug("[MD-CACHE][SELF] publish assembled failed: %s", e)

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
            agent_metadata = self._coerce_agent_metadata(data.get("agent_metadata"))
            rows = data.get("kv_caches_base_addr")

            # 如果 canonical 里还是 2D（len==num_layers），尝试用 part 聚合一次
            is_2d = (
                isinstance(rows, list)
                and len(rows) == int(self.num_layers)
                and rows
                and isinstance(rows[0], list)
                and len(rows[0]) == int(self.num_cache_entries)
                and all(isinstance(x, int) for x in rows[0])
            )
            if agent_tp > 1 and is_2d:
                assembled = self._try_assemble_md_from_parts(engine_id, agent_tp)
                if assembled:
                    agent_metadata = self._coerce_agent_metadata(assembled["agent_metadata"])
                    rows = assembled["kv_caches_base_addr"]
                    # 顺便修正 canonical，避免下一次还读到 2D
                    try:
                        self._persist_remote_md_cache(
                            engine_id=engine_id,
                            agent_metadata=agent_metadata,
                            kv_caches_base_addr=rows,
                            num_blocks=int(assembled["num_blocks"]),
                            kv_caches_dev_ids=assembled.get("kv_caches_dev_ids"),
                            agent_tp=int(assembled["agent_tp"]),
                        )
                        logger.info("[MD-CACHE] repaired canonical md for engine=%s via parts", engine_id)
                    except Exception:
                        pass

            kv_rows_norm = self._normalize_kv_rows(engine_id, rows, agent_tp)
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

    # -------------------- 你原来的 read_blocks / write_blocks / add_remote_agent 等保持不变 --------------------
    # 下面只贴关键：register_kv_caches 里 publish self md 改成 _publish_self_md_cache

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

        self._le_list_cache = None

        # ✅ PATCH: self publish 改成 part->assemble->canonical
        if _env_flag("NIXL_PUBLISH_SELF_MD", True):
            try:
                self._publish_self_md_cache()
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

    @staticmethod
    def _peek(xs, k=3):
        return xs[:k] + (["..."] if len(xs) > k else [])

    def _normalize_kv_rows(self, engine_id: str, rows, agent_tp: int):
        """
        允许两种输入：
          - 三维: [tp][layers][entries]  -> 直接返回
          - 二维: [layers][entries] 且 agent_tp==1 -> 自动包一层变成 [1][layers][entries]
        其它形状一律报错。
        """
        if rows is None:
            return []

        if (isinstance(rows, list) and rows and isinstance(rows[0], list)
                and len(rows) == int(agent_tp) and isinstance(rows[0][0], (int,)) is False):
            return rows

        if (int(agent_tp) == 1 and isinstance(rows, list)
                and len(rows) == int(self.num_layers)
                and all(isinstance(x, list) and len(x) == int(self.num_cache_entries) for x in rows)):
            return [rows]

        raise RuntimeError(
            f"[ADD] kv_caches_base_addr shape incompatible with agent_tp={agent_tp}: "
            f"outer_len={len(rows) if isinstance(rows, list) else 'NA'} "
            f"(expect {agent_tp} or {self.num_layers} when tp==1)"
        )

    # read_blocks / write_blocks / add_remote_agent / get_done_transfers / get_timing
    # 这一大段你原样保留即可（你贴的版本就行）。


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
        self._ns = defaultdict(int)
        self._n = defaultdict(int)
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
