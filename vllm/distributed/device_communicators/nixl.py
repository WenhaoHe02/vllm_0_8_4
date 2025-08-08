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
from typing import List, Tuple
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
        # required for @cached_property.
        dict=True):
    engine_id: str
    agent_metadata: List[bytes]
    kv_caches_base_addr: List[List[List[int]]] # base address for each rank for each layer for keys and values
    num_blocks: int


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

        self._transfers = defaultdict(list)


        self._tp_size[engine_id] = vllm_config.parallel_config.tensor_parallel_size
        self._is_mla = "deepseek" in vllm_config.model_config.architectures[0].lower()

        self.block_size = None
        self.head_dim = None

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
                kv_caches_base_addr.append([base_addr,])

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
        # This function should return a list of ranges of block ids that are contiguous
        # For example, if block_ids is [0, 1, 2, 4, 5, 6], the function should return [[0, 2], [4, 6]]
        # The ranges are sorted by the starting block id
        # The function should also make sure that the block ids are contiguous
        # If the block ids are not contiguous, the function should raise an error
        ranges = []
        for i in range(len(block_ids)):
            if i == 0 or block_ids[i] != block_ids[i-1] + 1:
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
                            descs_ids.append(layer_id * self.num_cache_entries * num_blocks * tp_multiplier + entry_index * num_blocks * tp_multiplier + start_offset * tp_multiplier + i_offset + (block_id - start_offset))
                        else:
                            descs_ids.append(layer_id * self.num_cache_entries * num_blocks + entry_index * num_blocks + block_id)
        else:
            num_blocks = self.dst_num_blocks[engine_id]
            for layer_id in layer_ids:
                for entry_index in range(self.num_cache_entries):
                    for block_id in block_ids:
                        descs_ids.append(layer_id * self.num_cache_entries * num_blocks + entry_index * num_blocks + block_id)
        return descs_ids

    def _get_same_length_ranges(self, src_ranges, dst_ranges, return_original_src_ranges=False):
        # This function should return a list of ranges for both src and dst so that corresponding ranges are the same length
        # For example, if src_ranges is [[0, 2] [4, 8]] and dst_ranges is [[1, 3], [5, 7], [9, 10]]
        # The function should return ([[0, 2], [4, 6], [7, 8]], [[1, 3], [5, 7], [9, 10]])
        src_overlapping_ranges, dst_overlapping_ranges = [], []

        original_src_ranges = []
        org_src_range = tuple(src_ranges[0])
        
        src_idx, dst_idx = 0, 0
        while src_idx < len(src_ranges) and dst_idx < len(dst_ranges):
            src_range = src_ranges[src_idx]
            dst_range = dst_ranges[dst_idx]
            
            # Calculate the length of each range
            src_len = src_range[-1] - src_range[0] + 1
            dst_len = dst_range[-1] - dst_range[0] + 1
            
            # If ranges have the same length, add them directly
            if src_len == dst_len:
                src_overlapping_ranges.append([src_range[0], src_range[-1]])
                dst_overlapping_ranges.append([dst_range[0], dst_range[-1]])
                original_src_ranges.append(org_src_range)
                src_idx += 1
                dst_idx += 1
                if src_idx < len(src_ranges):
                    org_src_range = tuple(src_ranges[src_idx])
            # If source range is longer, split it
            elif src_len > dst_len:
                src_overlapping_ranges.append([src_range[0], src_range[0] + dst_len - 1])
                dst_overlapping_ranges.append([dst_range[0], dst_range[-1]])
                original_src_ranges.append(org_src_range)
                # Update source range for next iteration
                src_ranges[src_idx] = [src_range[0] + dst_len, src_range[-1]]
                dst_idx += 1
            # If destination range is longer, split it
            else:  # src_len < dst_len
                src_overlapping_ranges.append([src_range[0], src_range[-1]])
                dst_overlapping_ranges.append([dst_range[0], dst_range[0] + src_len - 1])
                original_src_ranges.append(org_src_range)
                # Update destination range for next iteration
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
            # TODO ptarasiewicz: we skip staging when is_mla is true, we shouldn't assign staging blocks at all
            staging_rearranging_ranges = None
            staging_block_ids = local_block_ids
        else:
            local_ranges = self._get_ranges(local_block_ids)
            staging_ranges = self._get_ranges(staging_block_ids)

            local_rearranging_ranges, staging_rearranging_ranges = self._get_same_length_ranges(local_ranges, staging_ranges)

        tp_multiplier = self._tp_size[dst_engine_id] // self._tp_size[self.engine_id]
        remote_block_descs_ids = self._get_block_descs_ids(dst_engine_id, "all", remote_block_ids)
        local_xfer_side_handle = self.src_xfer_side_handles[tp_multiplier]
        handles = []

        logger.debug("Time to get block descs ids: %s ms", (time.perf_counter() - start_time) * 1000)
        create_xfer_start_time = time.perf_counter()

        for i in range(tp_multiplier):
            staging_block_descs_ids = self._get_block_descs_ids(self.engine_id, "all", staging_block_ids, i=i, tp_multiplier=tp_multiplier, staging_ranges=staging_rearranging_ranges)
            assert len(staging_block_descs_ids) == len(remote_block_descs_ids)
            remote_xfer_side_handle = self.dst_xfer_side_handles[dst_engine_id][i]
            handle = self.nixl_wrapper.make_prepped_xfer("READ", local_xfer_side_handle, staging_block_descs_ids, 
                                                        remote_xfer_side_handle, remote_block_descs_ids, 
                                                        "")
            handles.append(handle)
            status = self.nixl_wrapper.transfer(handle)

        logger.debug("Time to create xfer: %s ms", (time.perf_counter() - create_xfer_start_time) * 1000)

        transfer_start_time = time.perf_counter()

        for handle in handles:
            while (status := self.nixl_wrapper.check_xfer_state(handle)) != "DONE":
                if status == "PROC":
                    time.sleep(0.001)
                else:
                    raise RuntimeError("Read transfer failed with state %s", status)
            # self.nixl_wrapper.abort_xfer(handle) # TODO ptarasiewicz: why abort is throwing errors?

        logger.debug("Time to transfer: %s ms", (time.perf_counter() - transfer_start_time) * 1000)

        rearrange_start_time = time.perf_counter()

        if not self._is_mla:
            for local_range, staging_range in zip(local_rearranging_ranges, staging_rearranging_ranges):
                logger.debug("Rearranging tensors for cache: %s, local_range: %s, staging_range: %s", self.kv_caches[0].shape, local_range, staging_range)
                for kv_cache in self.kv_caches:
                    for cache in kv_cache:
                        rearrange_tensors(cache[local_range[0]:local_range[1] + 1], cache[staging_range[0]:staging_range[1] + 1], tp_multiplier, "read")

        logger.debug("Time to rearrange tensors: %s ms", (time.perf_counter() - rearrange_start_time) * 1000)
        logger.debug("Total time for read: %s ms", (time.perf_counter() - start_time) * 1000)

    def write_blocks(self, local_block_ids, staging_block_ids, remote_block_ids, dst_engine_id, notify_msg):
        logger.debug("Writing %d blocks to %s from %s with notify message %s",
                     len(local_block_ids), dst_engine_id, self.agent_name, notify_msg)

        remote_block_ids = remote_block_ids[:len(local_block_ids)]
        assert len(staging_block_ids) == len(local_block_ids)
        tp_multiplier = self._tp_size[dst_engine_id] // self._tp_size[self.engine_id]

        downscale_info = self._downscale_info.get(dst_engine_id)
        force_serial = getattr(self.vllm_config.kv_transfer_config, "force_serial_write", False)
        serial_mode = (downscale_info is not None) or force_serial

        if serial_mode and downscale_info is not None:
            eff_tp = 1
            targets = [0]
        else:
            eff_tp = max(1, tp_multiplier)
            targets = list(range(eff_tp))

        if len(local_block_ids) == 0:
            logger.debug("No blocks to write")
            if downscale_info is not None:
                rr = downscale_info["remote_rank"]
                self.nixl_wrapper.send_notif(self._remote_agents[dst_engine_id][rr], notify_msg)
            else:
                for i in range(tp_multiplier):
                    self.nixl_wrapper.send_notif(
                        self._remote_agents[dst_engine_id][self.rank * tp_multiplier + i], notify_msg)
            return

        start_time = time.perf_counter()

        if self._is_mla:
            staging_rearranging_ranges = None
            staging_block_ids = local_block_ids
        else:
            local_ranges = self._get_ranges(local_block_ids)
            staging_ranges = self._get_ranges(staging_block_ids)
            local_rearranging_ranges, staging_rearranging_ranges = \
                self._get_same_length_ranges(local_ranges, staging_ranges)
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

        if serial_mode:
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
                    notify_msg
                )
                self.nixl_wrapper.transfer(handle)
                while True:
                    status = self.nixl_wrapper.check_xfer_state(handle)
                    if status == "DONE":
                        break
                    elif status == "PROC":
                        time.sleep(0.001)
                        continue
                    else:
                        raise RuntimeError(f"Write transfer failed with state {status}")

            logger.debug("Time to create & finish all xfers: %s ms",
                         (time.perf_counter() - create_xfer_start_time) * 1000)
            logger.debug("Total time for write: %s ms", (time.perf_counter() - start_time) * 1000)
            return

        logger.debug("Creating xfer handles")
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
                notify_msg
            )
            self._transfers[notify_msg].append(handle)
            self.nixl_wrapper.transfer(handle)

        logger.debug("Time to create xfer: %s ms", (time.perf_counter() - create_xfer_start_time) * 1000)
        logger.debug("Total time for write: %s ms", (time.perf_counter() - start_time) * 1000)

    def get_notifs(self):
        return self.nixl_wrapper.update_notifs()
    
    def get_new_notifs(self):
        return self.nixl_wrapper.get_new_notifs()

    def add_remote_agent(self, engine_id, agent_metadata, agent_tp, kv_caches_base_addr, num_blocks):
        self._tp_size[engine_id] = agent_tp

        agent_names = []
        for agent_meta in agent_metadata:
            agent_name = self.nixl_wrapper.add_remote_agent(agent_meta)
            agent_names.append(agent_name)
        self._remote_agents[engine_id] = agent_names
        self.kv_caches_base_addr[engine_id] = kv_caches_base_addr
        tp_multiplier = self._tp_size[engine_id] // self._tp_size[self.engine_id]
        if tp_multiplier == 0 and not self._is_mla:
            group_size = self._tp_size[self.engine_id] // self._tp_size[engine_id]
            remote_rank = self.rank // group_size
            seg_len = self.block_len
            full_len = seg_len * group_size
            peer_offset = (self.rank % group_size) * seg_len
            self._downscale_info[engine_id] = {"group_size": group_size, "remote_rank": remote_rank}
            src_blocks = []
            for layer in range(self.num_layers):
                for base in self.kv_caches_base_addr[self.engine_id][layer]:
                    for bid in range(self.num_blocks):
                        src_blocks.append((base + bid * seg_len, seg_len, self.rank))
            src_desc = self.nixl_wrapper.get_xfer_descs(src_blocks, "VRAM")
            self.src_xfer_side_handles[1] = self.nixl_wrapper.prep_xfer_dlist("", src_desc)

            dst_blocks = []
            for layer in range(self.num_layers):
                for rbase in self.kv_caches_base_addr[engine_id][remote_rank][layer]:
                    for bid in range(num_blocks):
                        dst_blocks.append((rbase + bid * full_len + peer_offset, seg_len, remote_rank))
            dst_desc = self.nixl_wrapper.get_xfer_descs(dst_blocks, "VRAM")
            self.dst_xfer_side_handles[engine_id][0] = self.nixl_wrapper.prep_xfer_dlist(
                agent_names[remote_rank], dst_desc
            )
            self.dst_num_blocks[engine_id] = num_blocks
            self._tp_size[engine_id] = self._tp_size[self.engine_id]
            return agent_names
        assert tp_multiplier > 0, f"Decode TP cannot be smaller than prefill TP, got {self._tp_size[engine_id]} and {self._tp_size[self.engine_id]}"

        logger.debug("Creating src xfer side handles for engine %s, tp_multiplier: %s", engine_id, tp_multiplier)
        if self._is_mla:
            dst_block_len = self.block_len
        else:
            dst_block_len = self.block_len // tp_multiplier
        if tp_multiplier not in self.src_xfer_side_handles:
            # create descs and xfer side handles
            blocks_data = []
            for layer_id in range(self.num_layers):
                for base_addr in self.kv_caches_base_addr[self.engine_id][layer_id]:
                    for block_id in range(self.num_blocks):
                            block_offset = block_id * self.block_len
                            for i in range(1 if self._is_mla else tp_multiplier):
                                tp_multiplier_offset = i * dst_block_len
                                blocks_data.append((base_addr + block_offset + tp_multiplier_offset, dst_block_len, self.rank))
            logger.debug("Created %s blocks for src engine %s and rank %s", len(blocks_data), self.engine_id, self.rank * tp_multiplier + i)
            descs = self.nixl_wrapper.get_xfer_descs(blocks_data, "VRAM")
            self.src_xfer_side_handles[tp_multiplier] = self.nixl_wrapper.prep_xfer_dlist("", descs)

        # create dst xfer side handles
        self.dst_num_blocks[engine_id] = num_blocks
        for i in range(tp_multiplier):
            blocks_data = []
            for layer_id in range(self.num_layers):
                for base_addr in self.kv_caches_base_addr[engine_id][self.rank * tp_multiplier + i][layer_id]:
                    for block_id in range(num_blocks):
                        block_offset = block_id * dst_block_len
                        blocks_data.append((base_addr + block_offset, dst_block_len, self.rank * tp_multiplier + i))
            logger.debug("Created %s blocks for dst engine %s and rank %s", len(blocks_data), engine_id, self.rank * tp_multiplier + i)
            descs = self.nixl_wrapper.get_xfer_descs(blocks_data, "VRAM")
            self.dst_xfer_side_handles[engine_id][i] = self.nixl_wrapper.prep_xfer_dlist(self._remote_agents[engine_id][self.rank * tp_multiplier + i], descs)

        return agent_names

    def get_done_tranfers(self) -> List[str]:
        done_req_ids = []
        for req_id, handles in self._transfers.items():
            running_reqs = []
            for handle in handles:
                xfer_state = self.nixl_wrapper.check_xfer_state(handle)
                if xfer_state == "DONE":
                    # self.nixl_wrapper.release_xfer_handle(handle) # TODO ptarasiewicz: why abort is throwing errors?
                    continue
                if xfer_state == "PROC":
                    running_reqs.append(handle)
                else:
                    raise RuntimeError("Transfer failed with state %s", xfer_state)
            if len(running_reqs) == 0:
                done_req_ids.append(req_id)
            else:
                self._transfers[req_id] = running_reqs
        return done_req_ids
