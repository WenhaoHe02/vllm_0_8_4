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


    def read_blocks(self, local_block_ids, staging_block_ids, remote_block_ids, dst_engine_id):
        logger.debug("Reading %d blocks from %s to %s", len(local_block_ids), self.agent_name, dst_engine_id)
        assert len(local_block_ids) == len(staging_block_ids) == len(remote_block_ids)
        if len(local_block_ids) == 0:
            logger.debug("No blocks to read")
            return

        start_time = time.perf_counter()
        if self._is_mla:
            staging_rearranging_ranges = None
            staging_block_ids = local_block_ids
        else:
            local_ranges = self._get_ranges(local_block_ids)
            staging_ranges = self._get_ranges(staging_block_ids)
            local_rearranging_ranges, staging_rearranging_ranges = self._get_same_length_ranges(local_ranges, staging_ranges)

        downscale_info = self._downscale_info.get(dst_engine_id)
        tp_multiplier = self._tp_size[dst_engine_id] // self._tp_size[self.engine_id]
        if downscale_info is not None:
            eff_tp = 1
            targets = [0]
        else:
            eff_tp = max(1, tp_multiplier)
            targets = list(range(eff_tp))

        remote_block_descs_ids = self._get_block_descs_ids(dst_engine_id, "all", remote_block_ids)
        local_xfer_side_handle = self.src_xfer_side_handles[eff_tp]
        handles = []

        logger.debug("Time to get block descs ids: %s ms", (time.perf_counter() - start_time) * 1000)
        create_xfer_start_time = time.perf_counter()

        for i in targets:
            staging_block_descs_ids = self._get_block_descs_ids(
                self.engine_id, "all", staging_block_ids, i=i, tp_multiplier=eff_tp,
                staging_ranges=staging_rearranging_ranges
            )
            assert len(staging_block_descs_ids) == len(remote_block_descs_ids)
            remote_xfer_side_handle = self.dst_xfer_side_handles[dst_engine_id][i]
            handle = self.nixl_wrapper.make_prepped_xfer(
                "READ",
                local_xfer_side_handle, staging_block_descs_ids,
                remote_xfer_side_handle, remote_block_descs_ids,
                ""
            )
            self.nixl_wrapper.transfer(handle)
            handles.append(handle)

        logger.debug("Time to create xfer: %s ms", (time.perf_counter() - create_xfer_start_time) * 1000)
        transfer_start_time = time.perf_counter()

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
                    raise RuntimeError(f"Read transfer failed with state {status}")
            pending = nxt
            if pending:
                time.sleep(0.001)

        logger.debug("Time to transfer: %s ms", (time.perf_counter() - transfer_start_time) * 1000)
        rearrange_start_time = time.perf_counter()

        if not self._is_mla:
            for local_range, staging_range in zip(local_rearranging_ranges, staging_rearranging_ranges):
                logger.debug("Rearranging tensors for cache: %s, local_range: %s, staging_range: %s",
                             self.kv_caches[0].shape, local_range, staging_range)
                for kv_cache in self.kv_caches:
                    for cache in kv_cache:
                        rearrange_tensors(
                            cache[local_range[0]:local_range[1] + 1],
                            cache[staging_range[0]:staging_range[1] + 1],
                            eff_tp, "read"
                        )

        logger.debug("Time to rearrange tensors: %s ms", (time.perf_counter() - rearrange_start_time) * 1000)
        logger.debug("Total time for read: %s ms", (time.perf_counter() - start_time) * 1000)

    def write_blocks(self, local_block_ids, staging_block_ids, remote_block_ids, dst_engine_id, notify_msg):
        logger.debug("Writing %d blocks to %s from %s with notify message %s",
                     len(local_block_ids), dst_engine_id, self.agent_name, notify_msg)

        remote_block_ids = remote_block_ids[:len(local_block_ids)]
        assert len(staging_block_ids) == len(local_block_ids)

        downscale_info = self._downscale_info.get(dst_engine_id)
        tp_multiplier = self._tp_size[dst_engine_id] // self._tp_size[self.engine_id]
        if downscale_info is not None:
            eff_tp = 1
            targets = [0]
        else:
            eff_tp = max(1, tp_multiplier)
            targets = list(range(eff_tp))

        notify_bytes = notify_msg if isinstance(notify_msg, (bytes, bytearray)) else str(notify_msg).encode()

        if len(local_block_ids) == 0:
            logger.debug("No blocks to write")
            if downscale_info is not None:
                rr = downscale_info["remote_rank"]
                self.nixl_wrapper.send_notif(self._remote_agents[dst_engine_id][rr], notify_bytes)
            else:
                for i in range(tp_multiplier):
                    self.nixl_wrapper.send_notif(
                        self._remote_agents[dst_engine_id][self.rank * tp_multiplier + i],
                        notify_bytes
                    )
            return

        start_time = time.perf_counter()
        if self._is_mla:
            staging_rearranging_ranges = None
            staging_block_ids = local_block_ids
        else:
            local_ranges = self._get_ranges(local_block_ids)
            staging_ranges = self._get_ranges(staging_block_ids)
            local_rearranging_ranges, staging_rearranging_ranges = self._get_same_length_ranges(local_ranges, staging_ranges)
            for local_range, staging_range in zip(local_rearranging_ranges, staging_rearranging_ranges):
                logger.debug("Rearranging tensors for cache: %s, local_range: %s, staging_range: %s",
                             self.kv_caches[0].shape, local_range, staging_range)
                for kv_cache in self.kv_caches:
                    for cache in kv_cache:
                        rearrange_tensors(
                            cache[local_range[0]: local_range[1] + 1],
                            cache[staging_range[0]: staging_range[1] + 1],
                            eff_tp, "write"
                        )

        logger.debug("Time to rearrange tensors: %s ms", (time.perf_counter() - start_time) * 1000)
        create_xfer_start_time = time.perf_counter()

        remote_block_descs_ids = self._get_block_descs_ids(dst_engine_id, "all", remote_block_ids)
        local_xfer_side_handle = self.src_xfer_side_handles[eff_tp]

        for i in targets:
            staging_block_descs_ids = self._get_block_descs_ids(
                self.engine_id, "all", staging_block_ids,
                i=i, tp_multiplier=eff_tp, staging_ranges=staging_rearranging_ranges
            )
            assert len(staging_block_descs_ids) == len(remote_block_descs_ids)
            remote_xfer_side_handle = self.dst_xfer_side_handles[dst_engine_id][i]
            handle = self.nixl_wrapper.make_prepped_xfer(
                "WRITE",
                local_xfer_side_handle, staging_block_descs_ids,
                remote_xfer_side_handle, remote_block_descs_ids,
                notify_bytes
            )
            self._transfers[notify_bytes].append(handle)
            self.nixl_wrapper.transfer(handle)

        logger.debug("Time to create xfer: %s ms", (time.perf_counter() - create_xfer_start_time) * 1000)
        logger.debug("Total time for write: %s ms", (time.perf_counter() - start_time) * 1000)

    def get_notifs(self):
        return self.nixl_wrapper.update_notifs()

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
        self._tp_size[engine_id] = agent_tp
        agent_names: List[str] = []
        for meta in agent_metadata:
            agent_names.append(self.nixl_wrapper.add_remote_agent(meta))
        self._remote_agents[engine_id] = agent_names
        self.kv_caches_base_addr[engine_id] = kv_caches_base_addr
        self.kv_caches_dev_ids[engine_id] = kv_caches_dev_ids if kv_caches_dev_ids is not None else None
        loc_base = self.kv_caches_base_addr[engine_id]
        loc_dev = self.kv_caches_dev_ids[engine_id]
        assert isinstance(agent_tp, int) and agent_tp > 0
        assert len(agent_metadata) == agent_tp
        assert len(loc_base) == agent_tp
        for r in range(agent_tp):
            assert len(loc_base[r]) == self.num_layers
            for layer in range(self.num_layers):
                assert len(loc_base[r][layer]) == self.num_cache_entries
        tp_multiplier = self._tp_size[engine_id] // self._tp_size[self.engine_id]
        if tp_multiplier == 0 and not self._is_mla:
            group_size = self._tp_size[self.engine_id] // self._tp_size[engine_id]
            remote_rank = self.rank // group_size
            seg_len = self.block_len
            full_len = seg_len * group_size
            peer_offset = (self.rank % group_size) * seg_len

            self._downscale_info[engine_id] = {"group_size": group_size, "remote_rank": remote_rank}

            local_dev_id = int(torch.cuda.current_device())
            src_blocks = []
            for layer in range(self.num_layers):
                for base in self.kv_caches_base_addr[self.engine_id][layer]:
                    for bid in range(self.num_blocks):
                        src_blocks.append((base + bid * seg_len, seg_len, local_dev_id))
            src_desc = self.nixl_wrapper.get_xfer_descs(src_blocks, "VRAM")
            self.src_xfer_side_handles[1] = self.nixl_wrapper.prep_xfer_dlist("", src_desc)

            dst_blocks = []
            for layer in range(self.num_layers):
                layer_bases = self.kv_caches_base_addr[engine_id][remote_rank][layer]
                for entry_idx, rbase in enumerate(layer_bases):
                    rdev = int(remote_rank)  # 与 up 分支同假设：rank == device ordinal
                    for bid in range(num_blocks):
                        dst_blocks.append((rbase + bid * full_len + peer_offset, seg_len, rdev))
            dst_desc = self.nixl_wrapper.get_xfer_descs(dst_blocks, "VRAM")
            self.dst_xfer_side_handles[engine_id][0] = self.nixl_wrapper.prep_xfer_dlist(
                self._remote_agents[engine_id][remote_rank], dst_desc
            )

            try:
                self.nixl_wrapper.make_connection(self._remote_agents[engine_id][remote_rank])
            except Exception:
                pass

            self.dst_num_blocks[engine_id] = num_blocks
            self._tp_size[engine_id] = self._tp_size[self.engine_id]
            return agent_names
        assert tp_multiplier > 0
        logger.debug("Creating src xfer side handles for engine %s, tp_multiplier: %s",
                     engine_id, tp_multiplier)
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
                self.nixl_wrapper.make_connection(self._remote_agents[engine_id][remote_rank])
            except Exception as e:
                logger.debug("make_connection(%s) failed (lazy connect will be used): %s",
                             self._remote_agents[engine_id][remote_rank], e)
        return agent_names

    def get_done_tranfers(self) -> List[str]:
        done_req_ids = []
        for req_id, handles in self._transfers.items():
            running_reqs = []
            for handle in handles:
                xfer_state = self.nixl_wrapper.check_xfer_state(handle)
                if xfer_state == "DONE":
                    continue
                if xfer_state == "PROC":
                    running_reqs.append(handle)
                else:
                    raise RuntimeError(f"Transfer failed with state {xfer_state}")
            if len(running_reqs) == 0:
                done_req_ids.append(req_id)
            else:
                self._transfers[req_id] = running_reqs
        return done_req_ids
