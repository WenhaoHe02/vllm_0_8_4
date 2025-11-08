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
import triton
import triton.language as tl
from vllm.logger import init_logger
logger = init_logger(__name__)
@triton.jit
def rearrange_kernel_read(
    t1_ptr,
    t2_ptr,
    N,
    B,
    H,
    C,
    d,
    tensor_subset_size,
    block_size,
    token_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    curr_n = offsets // block_size
    curr_b = offsets // token_size % B
    curr_h = offsets // C % H 
    curr_c = offsets % C

    src_pos = offsets

    tp_group = curr_h * d // H
    dst_h = curr_h % (H // d)
    tp_group_offset = curr_n * (block_size // d) + curr_b * (H // d) * C + dst_h * C + curr_c

    dst_pos = tensor_subset_size * tp_group + tp_group_offset
    
    tl.store(t1_ptr + src_pos, tl.load(t2_ptr + dst_pos))

@triton.jit
def rearrange_kernel_write(
    t1_ptr,
    t2_ptr,
    N,
    B,
    H,
    C,
    d,
    tensor_subset_size,
    block_size,
    token_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    curr_n = offsets // block_size
    curr_b = offsets // token_size % B
    curr_h = offsets // C % H 
    curr_c = offsets % C

    src_pos = offsets

    tp_group = curr_h * d // H
    dst_h = curr_h % (H // d)
    tp_group_offset = curr_n * (block_size // d) + curr_b * (H // d) * C + dst_h * C + curr_c

    dst_pos = tensor_subset_size * tp_group + tp_group_offset
    
    tl.store(t2_ptr + dst_pos, tl.load(t1_ptr + src_pos))
    


def rearrange_tensors(t1: torch.Tensor, t2: torch.Tensor, d: int, direction: str):
    N, B, H, C = t1.shape
    
    assert t2.shape == (N, B, H, C), "Destination tensor must have same shape as source"
    assert H % d == 0, "H must be divisible by d"

    block_size = B * H * C
    token_size = H * C
    tensor_size = N * block_size
    tensor_subset_size = tensor_size // d
    
    BLOCK_SIZE = 1024
    grid = ((N * B * H * C + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    if direction == "read":
        rearrange_kernel_read[grid](
            t1, t2,
            N, B, H, C,
            d,
            tensor_subset_size,
            block_size,
            token_size,
            BLOCK_SIZE=BLOCK_SIZE
        )
    elif direction == "write":
        rearrange_kernel_write[grid](
            t1, t2,
            N, B, H, C,
            d,
            tensor_subset_size,
            block_size,
            token_size,
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        raise ValueError(f"Invalid direction: {direction}")

# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl

@triton.jit
def rearrange_kernel_write_down(  # grouped -> standard（合并）
    t_std_ptr,                    # 目标：标准布局 (N,B,H,C)
    t_grp_ptr,                    # 来源：组段连续 (N,B,H,C)
    N, B, H, C,                   # 形状：block数、block_size、总KV头数、head_dim
    ngroups,                      # 分组数 = tp_prefill // tp_decode，要求 H % ngroups == 0
    tensor_subset_size,           # = N * B * (H//ngroups) * C
    elems_per_block,              # = B * H * C
    elems_per_token,              # = H * C
    BLOCK_SIZE: tl.constexpr
):
    pid  = tl.program_id(0)
    base = pid * BLOCK_SIZE
    off  = base + tl.arange(0, BLOCK_SIZE)

    total = N * elems_per_block
    mask  = off < total

    # 线性下标 -> (n,b,h,c)   （标准布局的坐标）
    n = off // elems_per_block
    b = (off // elems_per_token) % B
    h = (off // C) % H
    c = off % C

    Hper = H // ngroups

    # 该元素属于哪一组，以及组内的头下标
    g   = (h * ngroups) // H       # 0..ngroups-1
    hin = h % Hper                 # 0..Hper-1

    # 在“组段连续”布局中的线性位置：
    #   先按组拼接，每组内是 (n,b,hin,c) 的标准顺序
    off_in_seg = n * (B * Hper * C) + b * (Hper * C) + hin * C + c
    pos_grp    = g * tensor_subset_size + off_in_seg

    # 标准布局中的线性位置就是 off
    val = tl.load(t_grp_ptr + pos_grp, mask=mask, other=0)
    tl.store(t_std_ptr + off, val, mask=mask)


@triton.jit
def rearrange_kernel_read_down(   # standard -> grouped（打包/对称校验）
    t_std_ptr,                    # 来源：标准布局 (N,B,H,C)
    t_grp_ptr,                    # 目标：组段连续 (N,B,H,C)
    N, B, H, C,
    ngroups,
    tensor_subset_size,
    elems_per_block,
    elems_per_token,
    BLOCK_SIZE: tl.constexpr
):
    pid  = tl.program_id(0)
    base = pid * BLOCK_SIZE
    off  = base + tl.arange(0, BLOCK_SIZE)

    total = N * elems_per_block
    mask  = off < total

    # 标准布局坐标
    n = off // elems_per_block
    b = (off // elems_per_token) % B
    h = (off // C) % H
    c = off % C

    Hper = H // ngroups

    g   = (h * ngroups) // H
    hin = h % Hper

    off_in_seg = n * (B * Hper * C) + b * (Hper * C) + hin * C + c
    pos_grp    = g * tensor_subset_size + off_in_seg

    val = tl.load(t_std_ptr + off, mask=mask, other=0)
    tl.store(t_grp_ptr + pos_grp, val, mask=mask)


def rearrange_tensors_down(
    t_standard: torch.Tensor,     # 目的/来源 (N,B,H,C) 标准布局
    t_grouped:  torch.Tensor,     # 来源/目的 (N,B,H,C) 组段连续布局
    ngroups: int,                 # = tp_prefill // tp_decode
    direction: str,               # "write": grouped->standard（合并）
                                  # "read" : standard->grouped（打包）
    block_size_elements: int = 1024
):
    """
    约束：
      - t_standard.shape == t_grouped.shape == (N,B,H,C)
      - 两者必须 contiguous，且内存不重叠（非原地）
      - H % ngroups == 0，H 为合并后的 KV 物理总头数
    """
    assert t_standard.shape == t_grouped.shape and t_standard.ndim == 4, "shape must be (N,B,H,C) and equal"
    assert t_standard.is_contiguous() and t_grouped.is_contiguous(), "inputs must be contiguous"
    N, B, H, C = t_standard.shape
    assert ngroups > 0 and (H % ngroups == 0), f"H={H} must be divisible by ngroups={ngroups}"

    elems_per_block  = B * H * C
    elems_per_token  = H * C
    tensor_subset_sz = N * B * (H // ngroups) * C
    total_elems      = N * elems_per_block
    grid = ((total_elems + block_size_elements - 1) // block_size_elements,)

    if direction == "write":
        # grouped -> standard（合并）
        rearrange_kernel_write_down[grid](
            t_standard, t_grouped,
            N, B, H, C,
            ngroups,
            tensor_subset_sz,
            elems_per_block,
            elems_per_token,
            BLOCK_SIZE=block_size_elements
        )
    elif direction == "read":
        # standard -> grouped（打包）
        rearrange_kernel_read_down[grid](
            t_standard, t_grouped,
            N, B, H, C,
            ngroups,
            tensor_subset_sz,
            elems_per_block,
            elems_per_token,
            BLOCK_SIZE=block_size_elements
        )
    else:
        raise ValueError("direction must be 'write' or 'read'")

# rearrange_read_down.py
import torch
import triton
import triton.language as tl

@triton.jit
def _rearrange_kernel_read_down(
    t_std_ptr, t_grp_ptr,  # src: standard (N,B,H,C) -> dst: grouped (N,B,H,C)
    N, B, H, C,
    ngroups,               # ngroups = tp_prefill // tp_decode
    tensor_subset_size,    # = N * B * (H//ngroups) * C
    elems_per_block,       # = B * H * C
    elems_per_token,       # = H * C
    BLOCK_SIZE: tl.constexpr
):
    pid  = tl.program_id(0)
    base = pid * BLOCK_SIZE
    off  = base + tl.arange(0, BLOCK_SIZE)

    total = N * elems_per_block
    mask  = off < total

    n = off // elems_per_block
    b = (off // elems_per_token) % B
    h = (off // C) % H
    c = off % C

    Hper = H // ngroups
    g    = (h * ngroups) // H        # 0..ngroups-1
    hin  = h % Hper                  # 0..Hper-1

    off_in_seg = n * (B * Hper * C) + b * (Hper * C) + hin * C + c
    pos_grp    = g * tensor_subset_size + off_in_seg

    val = tl.load(t_std_ptr + off, mask=mask, other=0)
    tl.store(t_grp_ptr + pos_grp, val, mask=mask)

@triton.jit
def _rearrange_kernel_read_down(
    t_std_ptr, t_grp_ptr,  # src: standard (N,B,H,C) -> dst: grouped (N,B,H,C)
    N, B, H, C,
    ngroups,               # ngroups = tp_prefill // tp_decode
    tensor_subset_size,    # = N * B * (H//ngroups) * C
    elems_per_block,       # = B * H * C
    elems_per_token,       # = H * C
    BLOCK_SIZE: tl.constexpr
):
    pid  = tl.program_id(0)
    base = pid * BLOCK_SIZE
    off  = base + tl.arange(0, BLOCK_SIZE)

    total = N * elems_per_block
    mask  = off < total

    n = off // elems_per_block
    b = (off // elems_per_token) % B
    h = (off // C) % H
    c = off % C

    Hper = H // ngroups
    g    = (h * ngroups) // H        # 0..ngroups-1
    hin  = h % Hper                  # 0..Hper-1

    # 在 grouped（按组拼接）中的线性位置
    off_in_seg = n * (B * Hper * C) + b * (Hper * C) + hin * C + c
    pos_grp    = g * tensor_subset_size + off_in_seg

    val = tl.load(t_std_ptr + off, mask=mask, other=0)
    tl.store(t_grp_ptr + pos_grp, val, mask=mask)

def rearrange_tensors_read_down(
    t_standard: torch.Tensor,
    t_grouped:  torch.Tensor,
    ngroups: int,
    block_size_elements: int = 1024,
):
    assert t_standard.shape == t_grouped.shape and t_standard.ndim == 4
    assert t_standard.is_contiguous() and t_grouped.is_contiguous()
    N, B, H, C = t_standard.shape
    logger.debug(f"[DBG] read_down shapes N={N}, B={B}, H={H}, C={C}, ngroups={ngroups}")
    print(f"[DBG] read_down shapes N={N}, B={B}, H={H}, C={C}, ngroups={ngroups}")
    elems_per_block  = B * H * C
    elems_per_token  = H * C
    tensor_subset_sz = N * B * (H // ngroups) * C
    total = N * elems_per_block
    grid = ((total + block_size_elements - 1) // block_size_elements,)

    _rearrange_kernel_read_down[grid](
        t_standard, t_grouped,
        N, B, H, C,
        ngroups,
        tensor_subset_sz,
        elems_per_block,
        elems_per_token,
        BLOCK_SIZE=block_size_elements,
    )

@triton.jit
def _rearrange_kernel_write_down(
    t_std_ptr, t_grp_ptr, N, B, H, C,
    ngroups, tensor_subset_size, elems_per_block, elems_per_token,
    BLOCK_SIZE: tl.constexpr
):
    pid  = tl.program_id(0)
    base = pid * BLOCK_SIZE
    off  = base + tl.arange(0, BLOCK_SIZE)

    total = N * elems_per_block
    mask  = off < total

    # 标准布局坐标
    n = off // elems_per_block
    b = (off // elems_per_token) % B
    h = (off // C) % H
    c = off % C

    Hper = H // ngroups         # 每组头数
    g   = (h * ngroups) // H    # 该头属于第几组
    hin = h % Hper              # 组内头下标

    # grouped 线性位置：按组拼接，每组内部仍是标准次序 (n,b,hin,c)
    off_in_seg = n * (B * Hper * C) + b * (Hper * C) + hin * C + c
    pos_grp    = g * tensor_subset_size + off_in_seg

    val = tl.load(t_grp_ptr + pos_grp, mask=mask, other=0)
    tl.store(t_std_ptr + off, val, mask=mask)

def _grouped_to_standard(t_grouped: torch.Tensor, ngroups: int) -> torch.Tensor:
    """将 shape=(N,B,H,C) 的 grouped 连续段合并成 standard；返回新 tensor。"""
    assert t_grouped.ndim == 4 and ngroups > 0
    N, B, H, C = t_grouped.shape
    assert H % ngroups == 0, "H must be divisible by ngroups"
    elems_per_block  = B * H * C
    elems_per_token  = H * C
    tensor_subset_sz = N * B * (H // ngroups) * C
    total            = N * elems_per_block
    BLOCK            = 1024
    grid             = ((total + BLOCK - 1) // BLOCK,)

    t_standard = torch.empty_like(t_grouped)
    _rearrange_kernel_write_down[grid](
        t_standard, t_grouped,
        N, B, H, C, ngroups,
        tensor_subset_sz, elems_per_block, elems_per_token,
        BLOCK_SIZE=BLOCK
    )
    return t_standard
