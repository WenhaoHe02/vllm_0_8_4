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
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16)


def _pick_from_pool(pool: list[int], base_slot: int, worker_idx: int) -> int:
    if not pool:
        try:
            return int(torch.cuda.current_device())
        except Exception:
            return 0
    return int(pool[(base_slot + worker_idx) % len(pool)])


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

        self._le_list_cache = None

        self._engine_fingerprint = {}
        self._engine_agent_tp = {}

        self._engine_prepped = {}

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

    # --------- retry wrapper for prep_xfer_dlist ---------
    def _prep_dlist_retry(self, agent_name, descs, backends=None,
                         tries: int = 80, sleep_s: float = 0.02, sleep_max: float = 0.2):
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

    # --------- choose remote dev_id for VRAM desc ---------
    def _remote_devid(self, remote_rank: int, pool_idx: int) -> int:
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

    @lru_cache(maxsize=1024)
    def _expand_seq(self, start_block: int, n_blocks: int) -> Tuple[int, ...]:
        B = int(self.block_size)
        return tuple([t for b in range(start_block, start_block + n_blocks) for t in range(b * B, b * B + B)])

    def _expand_blocks_to_tokens(self, block_ids: List[int]) -> List[int]:
        with self._timing.span("expand_blocks_to_tokens"):
            if not block_ids:
                return []
            rngs = self._get_ranges(block_ids)
            out = []
            for a, b in rngs:
                out.extend(self._expand_seq(a, b - a + 1))
            return out

    def _ranges_key(self, block_ids: List[int]) -> tuple:
        rngs = self._get_ranges(block_ids)
        return tuple((int(a), int(b)) for a, b in rngs)

    def _expand_blocks_to_tokens_cached(self, block_ids: List[int]) -> List[int]:
        if not block_ids:
            return []
        B = int(self.block_size)
        rk = self._ranges_key(block_ids)
        return list(_expand_blocks_to_tokens_cached_key(rk, B))

    def _get_le_list(self):
        if self._le_list_cache is None:
            self._le_list_cache = [(L, E) for L in range(self.num_layers) for E in range(self.num_cache_entries)]
        return self._le_list_cache

    def _ensure_down_ready(self, dst_engine_id: str) -> None:
        def _have_all() -> bool:
            down = self._downscale_info.get(dst_engine_id)
            if down is None:
                return False
            rr = down.get("remote_rank")
            if 1 not in self.src_xfer_side_handles or self.src_xfer_side_handles[1] is None:
                return False
            if (dst_engine_id not in self.dst_xfer_side_handles or
                    rr not in self.dst_xfer_side_handles[dst_engine_id] or
                    self.dst_xfer_side_handles[dst_engine_id][rr] is None):
                return False
            if dst_engine_id not in self.dst_num_blocks:
                return False
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

    def _write_blocks_down(self, local_block_ids, remote_block_ids, dst_engine_id, notify_msg):
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
                    group_size, peer_idx, is_leader
                )

            use_fast_path = (group_size > 1) and hasattr(dist, "is_available") and dist.is_available() and dist.is_initialized()
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

                # ✅ 兜底通知（避免 piggyback 丢）
                if is_leader and notify_payload:
                    try:
                        ra = self._remote_agents[dst_engine_id][remote_rank]
                        payload_b = notify_payload if isinstance(notify_payload, bytes) else notify_payload.encode("utf-8")
                        self.nixl_wrapper.send_notif(ra, payload_b)
                        logger.info("[DOWN-NOTIF] explicit send_notif ok key=%s", str(notify_payload)[:32])
                    except Exception as e:
                        logger.warning("[DOWN-NOTIF] explicit send_notif failed: %s", e)

                return

            base_rank = (self.rank // group_size) * group_size
            ranks_group = list(range(base_rank, base_rank + group_size))
            pg = info.get("pg")
            if pg is None:
                try:
                    pg = dist.new_group(ranks=ranks_group)
                    info["pg"] = pg
                except Exception as e:
                    logger.warning("[DOWN] new_group failed (%s), fallback to default group", e)
                    pg = None

            ra_decode = self._remote_agents[dst_engine_id][remote_rank]

            CHUNK_BLKS = int(os.getenv("NIXL_DOWN_PACK_CHUNK_BLOCKS", "16"))
            if CHUNK_BLKS <= 0:
                CHUNK_BLKS = len(local_block_ids)

            for L in range(self.num_layers):
                for entry_idx in range(self.num_cache_entries):
                    rbase = int(self.kv_caches_base_addr[dst_engine_id][remote_rank][L][entry_idx])

                    pool_idx = _remote_pool_index_by_env_or_md(dst_engine_id, remote_rank, L, entry_idx)
                    rdev = self._remote_devid(remote_rank, pool_idx)

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
                            if pg is not None:
                                dist.all_gather(recv_list, pack, group=pg)
                            else:
                                dist.all_gather(recv_list, pack)

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

            # ✅ leader 保险：无论 piggyback 成功与否，再显式发一次 notif
            if is_leader and notify_payload:
                try:
                    ra = self._remote_agents[dst_engine_id][remote_rank]
                    payload_b = notify_payload if isinstance(notify_payload, bytes) else notify_payload.encode("utf-8")
                    self.nixl_wrapper.send_notif(ra, payload_b)
                    logger.info("[DOWN-NOTIF] explicit send_notif ok key=%s", str(notify_payload)[:32])
                except Exception as e:
                    logger.warning("[DOWN-NOTIF] explicit send_notif failed: %s", e)

            if is_leader:
                logger.info(
                    "[WRITE][DOWN-fast] leader rank=%d remote_rank=%d group=%s -> single-write per block/entry/layer",
                    self.rank, remote_rank, ranks_group)
            else:
                logger.info("[WRITE][DOWN-fast] follower rank=%d done (no decode write)", self.rank)

    def _read_blocks_down(self, local_block_ids, staging_block_ids, remote_block_ids, dst_engine_id):
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
                if CHUNK_BLKS > 0:
                    steps = range(l0, l1 + 1, CHUNK_BLKS)
                else:
                    steps = [l0]

                for start in steps:
                    end = min(start + (CHUNK_BLKS if CHUNK_BLKS > 0 else (l1 - l0 + 1)), l1 + 1)
                    s_start = s0 + (start - l0)
                    s_end = s_start + (end - start)

                    for kv_cache in self.kv_caches:
                        for cache in kv_cache:
                            t_std = cache[s_start:s_end].contiguous()
                            t_grp = cache[start:end].contiguous()
                            rearrange_tensors_read_down(t_std, t_grp, ngroups)
                            cache[start:end].copy_(t_grp)

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

    # -------------------- md cache --------------------
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

    def _coerce_agent_metadata(self, md) -> List[bytes]:
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

    def _persist_remote_md_cache(
            self,
            engine_id: str,
            agent_metadata,
            kv_caches_base_addr,
            num_blocks: int,
            kv_caches_dev_ids,
            agent_tp: int,
    ) -> None:
        try:
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
            logger.debug("[MD-CACHE] persisted for engine=%s path=%s size=%dB",
                         engine_id, path, len(b))
        except Exception as e:
            logger.debug("[MD-CACHE] persist failed: %s", e)

    def _normalize_kv_rows(self, engine_id: str, rows, agent_tp: int):
        """
        期望：
          - 3D: [tp][layers][entries]
          - 2D: [layers][entries] 仅允许 agent_tp==1（会自动包装成 [1][layers][entries]）
        """
        if rows is None:
            return []

        def is_2d(x):
            return (isinstance(x, list)
                    and len(x) == int(self.num_layers)
                    and all(isinstance(l, list)
                            and len(l) == int(self.num_cache_entries)
                            and all(isinstance(v, int) for v in l)
                            for l in x))

        def is_3d(x):
            return (isinstance(x, list)
                    and len(x) == int(agent_tp)
                    and all(is_2d(xi) for xi in x))

        if is_3d(rows):
            return rows
        if int(agent_tp) == 1 and is_2d(rows):
            return [rows]

        raise RuntimeError(
            f"[ADD] kv_caches_base_addr shape incompatible with agent_tp={agent_tp}: "
            f"outer_len={len(rows) if isinstance(rows, list) else 'NA'} "
            f"(expect {agent_tp} or {self.num_layers} when tp==1)"
        )

    def _adopt_remote_md_from_cache(self, engine_id: str) -> bool:
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

    # -------------------- barrier --------------------
    def _barrier_dir(self, dst_engine_id: str, notify_key: str, group_size: int) -> str:
        base = os.getenv("NIXL_BARRIER_DIR", "/dev/shm" if os.path.isdir("/dev/shm") else "/tmp")
        safe_engine = self._sanitize_key(dst_engine_id, 16)
        safe_key = self._sanitize_key(notify_key, 24)
        d = os.path.join(base, f"nixl_down_bar_{safe_engine}_{safe_key}_{group_size}")
        os.makedirs(d, exist_ok=True)
        return d

    def _barrier_mark_and_wait(self, dst_engine_id: str, notify_key: str,
                               group_size: int, peer_idx: int, is_leader: bool) -> None:
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
                        break
                    time.sleep(0.001)

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

    # -------------------- register / publish self-md --------------------
    @property
    def agent_name(self):
        return self.nixl_wrapper.name

    def get_agent_metadata(self):
        return self.nixl_wrapper.get_agent_metadata()

    def _publish_self_md(self):
        """
        ✅ 修复点：tp>1 时，self-md 必须写成 3D: [tp][layers][entries]，否则你会看到 outer_len=64 / agent_tp=2 炸。
        实现：如果 dist 初始化了，就 all_gather_object 收集各 TP rank 的 base_addr 与 agent_metadata，由 rank0 写入 cache。
        """
        if not _env_flag("NIXL_PUBLISH_SELF_MD", True):
            return

        tp = int(self._tp_size[self.engine_id])
        local_md = self._coerce_agent_metadata(self.get_agent_metadata())
        local_rows_2d = self.kv_caches_base_addr[self.engine_id]
        local_dev_ids = self.kv_caches_dev_ids.get(self.engine_id)

        if tp <= 1:
            self._persist_remote_md_cache(
                engine_id=self.engine_id,
                agent_metadata=local_md,
                kv_caches_base_addr=local_rows_2d,  # 2D OK for tp==1
                num_blocks=int(self.num_blocks),
                kv_caches_dev_ids=local_dev_ids,
                agent_tp=1,
            )
            logger.info("[MD-CACHE][SELF] published engine=%s tp=1", self.engine_id)
            return

        # tp>1：尽量 gather 成完整 3D 后由 rank0 写
        try:
            import torch.distributed as dist
            if not (dist.is_available() and dist.is_initialized()):
                logger.warning("[MD-CACHE][SELF] tp=%d but dist not initialized -> skip self-md publish", tp)
                return

            md_list = [None for _ in range(tp)]
            rows_list = [None for _ in range(tp)]
            devs_list = [None for _ in range(tp)]

            # local_md 是 List[bytes]，但对每个 rank 只应有 1 个 bytes；若多了也照样传
            dist.all_gather_object(md_list, local_md)
            dist.all_gather_object(rows_list, local_rows_2d)
            dist.all_gather_object(devs_list, local_dev_ids)

            if dist.get_rank() % tp == 0:
                # 尽量把 agent_metadata 变成 List[bytes] 长度 tp：取每个 rank 的第一个 bytes
                agent_md_tp = []
                for m in md_list:
                    mm = self._coerce_agent_metadata(m)
                    agent_md_tp.append(mm[0] if mm else b"")

                kv_rows_3d = []
                for r in rows_list:
                    kv_rows_3d.append(r)

                self._persist_remote_md_cache(
                    engine_id=self.engine_id,
                    agent_metadata=agent_md_tp,
                    kv_caches_base_addr=kv_rows_3d,  # ✅ 3D
                    num_blocks=int(self.num_blocks),
                    kv_caches_dev_ids=None,
                    agent_tp=tp,
                )
                logger.info("[MD-CACHE][SELF] published engine=%s tp=%d (rank0 of tp-group)", self.engine_id, tp)
        except Exception as e:
            logger.warning("[MD-CACHE][SELF] publish tp=%d failed: %s", tp, e)

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

        self._le_list_cache = None

        # ✅ 修复：tp>1 也能写出正确 3D self-md（避免 outer_len=64/agent_tp=2）
        self._publish_self_md()

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

    # -------------------- 原有 read_blocks / write_blocks / add_remote_agent（保留你逻辑，仅加两处关键修复） --------------------
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

    def read_blocks(self, local_block_ids, staging_block_ids, remote_block_ids, dst_engine_id):
        with self._timing.span("read_blocks"):
            logger.info("[READ] local=%s staging=%s remote=%s dst_engine=%s",
                        len(local_block_ids), len(staging_block_ids), len(remote_block_ids), dst_engine_id)
            assert len(local_block_ids) == len(staging_block_ids) == len(remote_block_ids), \
                f"[READ] len mismatch: local={len(local_block_ids)} staging={len(staging_block_ids)} remote={len(remote_block_ids)}"
            if len(local_block_ids) == 0:
                logger.info("[READ] no-op (0 blocks)")
                return

            if self._is_mla:
                staging_rearranging_ranges = None
                staging_block_ids = local_block_ids
            else:
                local_ranges = self._get_ranges(local_block_ids)
                staging_ranges = self._get_ranges(staging_block_ids)
                local_rearranging_ranges, staging_rearranging_ranges = self._get_same_length_ranges(local_ranges, staging_ranges)
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
                handle = self.nixl_wrapper.make_prepped_xfer(
                    "READ",
                    local_xfer_side_handle, staging_block_descs_ids,
                    remote_xfer_side_handle, remote_block_descs_ids,
                    ""
                )
                self.nixl_wrapper.transfer(handle)
                handles.append(handle)

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

                wait_ms = int(os.getenv("NIXL_READY_WAIT_MS", "3000"))
                t0 = time.time()
                last_missing = "unknown"

                while True:
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
                            try:
                                if self._adopt_remote_md_from_cache(dst_engine_id):
                                    logger.info("[WRITE][ADOPT] late-adopted metadata for dst=%s", dst_engine_id)
                                    continue
                            except Exception as _e:
                                logger.debug("[WRITE][ADOPT] late adopt failed for dst=%s: %s", dst_engine_id, _e)
                            last_missing = f"tp_size_missing(dst_has={tp_dst is not None}, src_has=True)"

                    if (time.time() - t0) * 1000.0 > wait_ms:
                        raise RuntimeError(
                            f"[WRITE] precondition not met on rank={self.rank} dst={dst_engine_id}: {last_missing} ; "
                            f"_tp_size_keys={list(self._tp_size.keys())} src_keys={list(self.src_xfer_side_handles.keys())} "
                            f"dst_keys_top={list(self.dst_xfer_side_handles.keys())}"
                        )
                    time.sleep(0.001)

                def _to_notify_str(x):
                    return x if isinstance(x, str) else str(x)

                # ---------------- DOWN ----------------
                if self._downscale_info.get(dst_engine_id) is not None:
                    info = self._downscale_info[dst_engine_id]
                    remote_rank = info["remote_rank"]

                    # ✅ 关键修复1：所有 rank 都把 req_id 标成 done（否则上层可能等全 rank done 而不返回）
                    notify_key = _to_notify_str(notify_msg)
                    if notify_key:
                        self._transfers.setdefault(notify_key, [])
                        self._transfers[notify_key] = []

                    if 1 not in self.src_xfer_side_handles or self.src_xfer_side_handles[1] is None:
                        raise RuntimeError(f"[WRITE] DOWN missing src handle (rank={self.rank})")
                    if (dst_engine_id not in self.dst_xfer_side_handles or
                            remote_rank not in self.dst_xfer_side_handles[dst_engine_id]):
                        raise RuntimeError(f"[WRITE] DOWN missing dst handle (rank={self.rank} rr={remote_rank})")

                    self._write_blocks_down(local_block_ids, remote_block_ids, dst_engine_id, notify_msg)

                    # 再保险：确保仍是空列表
                    if notify_key:
                        self._transfers[notify_key] = []

                    logger.info("[WRITE] end ok dst=%s (DOWN)", dst_engine_id)
                    if self._timing_autolog:
                        stats = self.get_timing(reset=True)
                        if stats:
                            logger.info("[TIMING][WRITE-DOWN] %s", stats)
                    return

                # ---------------- UP/EQ ----------------
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

    def get_timing(self, reset: bool = False):
        stats = self._timing.snapshot(reset=reset)
        if stats:
            logger.debug("[TIMING] %s", stats)
        return stats

    # -------------------- add_remote_agent（按你原版保留；你之前贴的内容太长，这里不再重复粘一遍） --------------------
    # 你把你现有 add_remote_agent 原样放回这里即可（我上面用到的 _prep_dlist_retry / _normalize_kv_rows / self-md 已经修复）。
    #
    # ⚠️ 重要：如果你要我把 add_remote_agent 也“逐行合并成一个最终版本”，
    # 你只需要回复一句“把 add_remote_agent 也合进去”，我会把它也完整贴出来，
    # 但你当前这份对话里 add_remote_agent 已经在你原文件里存在且很长，我避免重复刷屏。
    #
    # -------------------- END --------------------
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

            self._check_engine_id_reuse(engine_id, agent_metadata, agent_tp)

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

            self.kv_caches_base_addr[engine_id] = kv_caches_base_addr
            self.kv_caches_dev_ids[engine_id] = kv_caches_dev_ids if kv_caches_dev_ids is not None else None
            loc_base = self.kv_caches_base_addr[engine_id]

            if len(agent_metadata) != agent_tp:
                raise RuntimeError(f"[ADD] agent_metadata len={len(agent_metadata)} != agent_tp={agent_tp}")
            if len(loc_base) != agent_tp:
                raise RuntimeError(f"[ADD] kv_caches_base_addr outer len={len(loc_base)} != agent_tp={agent_tp}")
            for r in range(agent_tp):
                assert len(loc_base[r]) == self.num_layers
                for L in range(self.num_layers):
                    assert len(loc_base[r][L]) == self.num_cache_entries

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
                devs = self.kv_caches_dev_ids.get(r_engine_id)
                if devs is not None:
                    try:
                        v = int(devs[r_idx][layer][entry_idx])
                        return v
                    except Exception:
                        logger.warning("[ADD] invalid kv_caches_dev_ids for engine=%s r=%d L=%d E=%d",
                                       r_engine_id, r_idx, layer, entry_idx)
                if r_idx in _env_map:
                    v = int(_env_map[r_idx])
                    if _pool_len_hint and v >= _pool_len_hint:
                        logger.warning("[MAP] %s maps %d->%d out of pool_len=%d",
                                       ENV_MAP_NAME, r_idx, v, _pool_len_hint)
                    return v
                if _pool_len_hint:
                    v = r_idx % _pool_len_hint
                    logger.info("[MAP][FALLBACK] %s not set for %d, fallback pool_index=%d (pool_len=%d)",
                                ENV_MAP_NAME, r_idx, v, _pool_len_hint)
                    return v
                logger.info("[MAP][FALLBACK] %s empty and pool_len unknown; use 0", ENV_MAP_NAME)
                return 0

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
                    "[ADD][DOWN] group_size=%d remote_rank=%d peer_idx=%d token_len_local=%d token_len_total=%d full_len=%d seg_len=%d peer_off_tok=%d",
                    group_size, remote_rank, peer_idx, token_len_local, token_len_total, full_len, seg_len,
                    peer_off_tok)

                BACKENDS = ["UCX"] if os.getenv("NIXL_FORCE_UCX", "1") == "1" else None

                if 1 not in self.src_xfer_side_handles or self.src_xfer_side_handles[1] is None:
                    src_blocks = []
                    local_dev_id = self.rank
                    for layer in range(self.num_layers):
                        for base in self.kv_caches_base_addr[self.engine_id][layer]:
                            for bid in range(self.num_blocks):
                                base_block = base + bid * seg_len
                                for t in range(B):
                                    src_blocks.append((base_block + t * token_len_local, token_len_local, local_dev_id))
                    desc = self.nixl_wrapper.get_xfer_descs(src_blocks, "VRAM")
                    self.src_xfer_side_handles[1] = self.nixl_wrapper.prep_xfer_dlist("", desc, backends=BACKENDS)

                if engine_id not in self.dst_xfer_side_handles:
                    self.dst_xfer_side_handles[engine_id] = {}
                if remote_rank not in self.dst_xfer_side_handles[engine_id]:
                    dst_blocks = []
                    for layer in range(self.num_layers):
                        layer_bases = self.kv_caches_base_addr[engine_id][remote_rank][layer]
                        for entry_idx, rbase in enumerate(layer_bases):
                            pool_idx = _remote_pool_index_by_env_or_md(engine_id, remote_rank, layer, entry_idx)
                            rdev = self._remote_devid(remote_rank, pool_idx)  # ✅ default rank
                            for bid in range(num_blocks):
                                base_block = rbase + bid * full_len
                                for t in range(B):
                                    dst_blocks.append((base_block + t * token_len_total + peer_off_tok,
                                                       token_len_local, rdev))
                    desc = self.nixl_wrapper.get_xfer_descs(dst_blocks, "VRAM")
                    # ✅ retry NOT_FOUND
                    self.dst_xfer_side_handles[engine_id][remote_rank] = self._prep_dlist_retry(
                        self._remote_agents[engine_id][remote_rank],
                        desc,
                        backends=BACKENDS,
                    )
                    try:
                        self.nixl_wrapper.make_connection(self._remote_agents[engine_id][remote_rank])
                    except Exception as e:
                        logger.debug("make_connection lazy: %s", e)

                if "read_down_src" not in self.src_xfer_side_handles or self.src_xfer_side_handles[
                    "read_down_src"] is None:
                    B = int(self.block_size)
                    H_loc = int(self.num_heads)
                    C = int(self.head_dim)
                    e = self.kv_caches[0][0].element_size()
                    token_len_local = H_loc * C * e
                    seg_len_local = B * token_len_local

                    blocks_local = []
                    local_dev_id = self.rank
                    for layer in range(self.num_layers):
                        for base in self.kv_caches_base_addr[self.engine_id][layer]:
                            for bid in range(self.num_blocks):
                                base_block = base + bid * seg_len_local
                                for t in range(B):
                                    blocks_local.append(
                                        (base_block + t * token_len_local, token_len_local, local_dev_id))
                    descs_local = self.nixl_wrapper.get_xfer_descs(blocks_local, "VRAM")
                    self.src_xfer_side_handles["read_down_src"] = self.nixl_wrapper.prep_xfer_dlist("", descs_local,
                                                                                                    backends=BACKENDS)

                if "read_down_dst" not in self.dst_xfer_side_handles[engine_id]:
                    B = int(self.block_size)
                    H_loc = int(self.num_heads)
                    C = int(self.head_dim)
                    e = self.kv_caches[0][0].element_size()
                    token_len_local = H_loc * C * e
                    token_len_total = group_size * token_len_local
                    seg_len_total = B * token_len_total
                    peer_off = peer_idx * token_len_local

                    blocks_remote = []
                    for layer in range(self.num_layers):
                        layer_bases = self.kv_caches_base_addr[engine_id][remote_rank][layer]
                        for entry_idx, rbase in enumerate(layer_bases):
                            pool_idx = _remote_pool_index_by_env_or_md(engine_id, remote_rank, layer, entry_idx)
                            rdev = self._remote_devid(remote_rank, pool_idx)  # ✅ default rank
                            for bid in range(num_blocks):
                                base_block = rbase + bid * seg_len_total
                                for t in range(B):
                                    blocks_remote.append((base_block + t * token_len_total + peer_off,
                                                          token_len_local, rdev))
                    descs_remote = self.nixl_wrapper.get_xfer_descs(blocks_remote, "VRAM")
                    # ✅ retry NOT_FOUND
                    self.dst_xfer_side_handles[engine_id]["read_down_dst"] = self._prep_dlist_retry(
                        self._remote_agents[engine_id][remote_rank],
                        descs_remote,
                        backends=BACKENDS,
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
                return self._remote_agents[engine_id]

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


def _env_flag(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip() not in ("0", "false", "False", "")


class _Timing:
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
