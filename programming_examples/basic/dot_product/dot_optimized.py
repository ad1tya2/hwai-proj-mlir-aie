#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

import numpy as np
import sys
import argparse

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2, Tile
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron.controlflow import range_

def my_dot_optimized(dev, num_elements, trace_size):
    # Architecture: 2 Columns, 8 Tiles
    # Col 0: Tiles (0, 2), (0, 3), (0, 4), (0, 5)
    # Col 1: Tiles (1, 2), (1, 3), (1, 4), (1, 5)
    
    num_tiles = 8
    
    if num_elements % num_tiles != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be divisible by {num_tiles}."
        )
        
    chunk_size = num_elements // num_tiles
    
    # We will use a tile_size for the kernel loop. 
    # For 2560/8 = 320, 6912/8 = 864. 
    # These fit in a single buffer, but let's stick to a standard tile size if possible or just use chunk_size.
    # Let's use chunk_size as the tile size for simplicity since they are small.
    tile_size = chunk_size
    
    dtype = np.int32

    # Define tensor types
    tensor_in_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    # Output: 1 partial sum per tile -> 8 partial sums total
    # We will collect 4 on Shim 0 and 4 on Shim 1
    tensor_out_ty = np.ndarray[(4,), np.dtype[dtype]] 
    
    tile_in_ty = np.ndarray[(tile_size,), np.dtype[dtype]]
    tile_out_ty = np.ndarray[(1,), np.dtype[dtype]]

    # --- Object FIFOs ---
    
    # Input A (in1) on Shim 0 (Tile 0, 0)
    # Distributed to all 8 tiles
    of_in1 = ObjectFifo(tile_in_ty, name="in1")
    
    # Input B (in2) on Shim 1 (Tile 1, 0)
    # Distributed to all 8 tiles
    of_in2 = ObjectFifo(tile_in_ty, name="in2")
    
    # Output 0 on Shim 0 (Tile 0, 0) - Collects from Col 0 tiles
    of_out0 = ObjectFifo(tile_out_ty, name="out0")
    
    # Output 1 on Shim 1 (Tile 1, 0) - Collects from Col 1 tiles
    of_out1 = ObjectFifo(tile_out_ty, name="out1")

    # --- Kernel ---
    dot_kernel = Kernel(
        "dot_product_int32_scalar", "dot.o", [tile_in_ty, tile_in_ty, tile_out_ty]
    )

    # --- Workers ---
    
    # List of compute tiles
    # Col 0 tiles
    compute_tiles_col0 = [Tile(0, r) for r in range(2, 6)]
    # Col 1 tiles
    compute_tiles_col1 = [Tile(1, r) for r in range(2, 6)]
    all_compute_tiles = compute_tiles_col0 + compute_tiles_col1
    
    # Split in1 to all 8 tiles
    # We need to define the split order. Let's say we split to Col 0 then Col 1.
    # offsets for split
    # Since we are splitting the whole tensor into 8 chunks of tile_size
    # The offsets are 0, tile_size, 2*tile_size ...
    # But ObjectFifo split takes a list of offsets into the PARENT fifo?
    # Actually split() creates consumer ports.
    # We need to specify how the data stream is split.
    # For a simple 1D split:
    
    # --- L2 ObjectFifos ---
    # We define these as the main buffers. 
    # rt.fill will write to them (Shim -> L2)
    # rt.drain will read from them (L2 -> Shim)
    # split/join will operate on them (L2 <-> L1)
    
    # in1 L2
    of_in1_L2 = ObjectFifo(tile_in_ty, name="in1_L2")
    
    # in2 L2
    of_in2_L2 = ObjectFifo(tile_in_ty, name="in2_L2")
    
    # out0 L2
    of_out0_L2 = ObjectFifo(tile_out_ty, name="out0_L2")
    
    # out1 L2
    of_out1_L2 = ObjectFifo(tile_out_ty, name="out1_L2")

    # --- Split/Join ---
    
    # in1 consumers (split from L2)
    # We place the split operation on Tile(0, 1)
    of_in1_cons = of_in1_L2.cons().split(
        [tile_size * i for i in range(num_tiles)], 
        obj_types=[tile_in_ty] * num_tiles,
        names=[f"in1_cons_{i}" for i in range(num_tiles)],
        placement=Tile(0, 1)
    )
    
    # in2 consumers (split from L2)
    # We place the split operation on Tile(1, 1)
    of_in2_cons = of_in2_L2.cons().split(
        [tile_size * i for i in range(num_tiles)],
        obj_types=[tile_in_ty] * num_tiles,
        names=[f"in2_cons_{i}" for i in range(num_tiles)],
        placement=Tile(1, 1)
    )
    
    # out0 producers (join to L2)
    # We place the join operation on Tile(0, 1)
    of_out0_L2_prod = of_out0_L2.prod().join(
        [1 * i for i in range(4)], 
        obj_types=[tile_out_ty] * 4,
        names=[f"out0_prod_{i}" for i in range(4)],
        placement=Tile(0, 1)
    )
    
    # out1 producers (join to L2)
    # We place the join operation on Tile(1, 1)
    of_out1_L2_prod = of_out1_L2.prod().join(
        [1 * i for i in range(4)],
        obj_types=[tile_out_ty] * 4,
        names=[f"out1_prod_{i}" for i in range(4)],
        placement=Tile(1, 1)
    )

    def core_body(in1, in2, out, kernel_func):
        # Process 1 chunk
        elem_in1 = in1.acquire(1)
        elem_in2 = in2.acquire(1)
        elem_out = out.acquire(1)
        kernel_func(elem_in1, elem_in2, elem_out)
        in1.release(1)
        in2.release(1)
        out.release(1)

    workers = []
    # Create workers for Col 0
    for i in range(4):
        workers.append(Worker(
            core_body,
            [of_in1_cons[i].cons(), of_in2_cons[i].cons(), of_out0_L2_prod[i].prod(), dot_kernel],
            placement=compute_tiles_col0[i]
        ))
        
    # Create workers for Col 1
    for i in range(4):
        workers.append(Worker(
            core_body,
            [of_in1_cons[i+4].cons(), of_in2_cons[i+4].cons(), of_out1_L2_prod[i].prod(), dot_kernel],
            placement=compute_tiles_col1[i]
        ))

    # --- Runtime ---
    # TAPs
    # in1: 8 chunks.
    tap_in1 = TensorAccessPattern((1, num_elements), 0, [1, 1, 1, num_elements], [0, 0, 0, 1])
    # in2: 8 chunks.
    tap_in2 = TensorAccessPattern((1, num_elements), 0, [1, 1, 1, num_elements], [0, 0, 0, 1])
    
    # out0: 4 elements
    tap_out0 = TensorAccessPattern((1, 4), 0, [1, 1, 1, 4], [0, 0, 0, 1])
    # out1: 4 elements
    tap_out1 = TensorAccessPattern((1, 4), 0, [1, 1, 1, 4], [0, 0, 0, 1])

    rt = Runtime()
    with rt.sequence(tensor_in_ty, tensor_in_ty, tensor_out_ty, tensor_out_ty) as (A, B, C0, C1):
        rt.start(*workers)
        tg = rt.task_group()
        
        # Fill in1 (A) -> L2
        rt.fill(of_in1_L2.prod(), A, tap_in1, task_group=tg, placement=Tile(0, 0))
        
        # Fill in2 (B) -> L2
        rt.fill(of_in2_L2.prod(), B, tap_in2, task_group=tg, placement=Tile(1, 0))
        
        # Drain out0 (C0) <- L2
        rt.drain(of_out0_L2.cons(), C0, tap_out0, wait=True, task_group=tg, placement=Tile(0, 0))
        
        # Drain out1 (C1) <- L2
        rt.drain(of_out1_L2.cons(), C1, tap_out1, wait=True, task_group=tg, placement=Tile(1, 0))
        
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", help="Device name", default="npu")
    parser.add_argument("--size", type=int, default=2560, help="Vector size")
    args = parser.parse_args()
    
    if args.device == "npu":
        dev = NPU1()
    elif args.device == "npu2":
        dev = NPU2()
    else:
        dev = NPU1()

    module = my_dot_optimized(dev, args.size, 0)
    print(module)
