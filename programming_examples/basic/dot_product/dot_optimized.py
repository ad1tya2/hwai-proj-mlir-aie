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
    # We use 4 MemTiles to distribute the load and avoid channel exhaustion.
    # L2(0,1) and L2(2,1) handle in1.
    # L2(1,1) and L2(3,1) handle in2.
    
    # in1 L2 primary (0,1)
    of_in1_L2_0 = ObjectFifo(tile_in_ty, name="in1_L2_0")
    # in1 L2 secondary (2,1)
    of_in1_L2_2 = ObjectFifo(tile_in_ty, name="in1_L2_2")
    
    # in2 L2 primary (1,1)
    of_in2_L2_1 = ObjectFifo(tile_in_ty, name="in2_L2_1")
    # in2 L2 secondary (3,1)
    of_in2_L2_3 = ObjectFifo(tile_in_ty, name="in2_L2_3")
    
    # out0 L2 (0,1)
    of_out0_L2 = ObjectFifo(tile_out_ty, name="out0_L2")
    # out1 L2 (1,1)
    of_out1_L2 = ObjectFifo(tile_out_ty, name="out1_L2")

    # --- Routing ---
    
    # in1: Shim 0 -> L2(0,1) -> L2(2,1)
    # We need to split in1 at L2(0,1) into "Col 0 part" and "Col 1 part".
    # Col 0 part stays in L2(0,1) and is distributed.
    # Col 1 part goes to L2(2,1) and is distributed.
    
    # We define the flow:
    # of_in1_L2_0 is the entry point.
    of_in1_L2_0 = of_in1.cons().forward(tile_in_ty, name="in1_L2_0", placement=Tile(0, 1))
    
    # Split at L2(0,1) into 2 branches: local distribution and forwarding
    # But wait, split creates consumers.
    # We want one consumer to be the distribution to Col 0, and another to be L2(2,1).
    # But distribution to Col 0 is 4 consumers.
    # So we split into 5 consumers? 4 for Col 0, 1 for L2(2,1).
    # Yes.
    
    # in1 split at L2(0,1)
    # Chunks 0..3 go to Col 0 tiles.
    # Chunks 4..7 go to L2(2,1).
    # We need to group chunks 4..7 into one stream for L2(2,1)? 
    # Or just send 4 streams?
    # If we send 4 streams, we use 4 channels.
    # We want to send 1 stream (chunks 4..7) to L2(2,1).
    # IRON `split` might not support hierarchical splitting easily (grouping chunks).
    # But we can split into 5: [chunk0, chunk1, chunk2, chunk3, chunk4_7].
    # But chunk4_7 must be contiguous?
    # The input stream is [0, 1, 2, 3, 4, 5, 6, 7].
    # We can split into [0, 1, 2, 3, 4..7].
    # Yes, by specifying offsets/sizes.
    
    # in1 consumers at L2(0,1)
    # 4 for Col 0 tiles, 1 for L2(2,1)
    of_in1_L2_0_cons = of_in1_L2_0.cons().split(
        [tile_size * 0, tile_size * 1, tile_size * 2, tile_size * 3, tile_size * 4],
        obj_types=[tile_in_ty, tile_in_ty, tile_in_ty, tile_in_ty, np.ndarray[(4*tile_size,), np.dtype(dtype)]],
        names=[f"in1_col0_{i}" for i in range(4)] + ["in1_to_L2_2"],
        placement=Tile(0, 1)
    )
    
    in1_col0_cons = of_in1_L2_0_cons[0:4]
    in1_to_L2_2 = of_in1_L2_0_cons[4]
    
    # Forward to L2(2,1)
    of_in1_L2_2 = in1_to_L2_2.forward(np.ndarray[(4*tile_size,), np.dtype(dtype)], name="in1_L2_2", placement=Tile(2, 1))
    
    # Split at L2(2,1) to Col 1 tiles
    of_in1_L2_2_cons = of_in1_L2_2.cons().split(
        [tile_size * i for i in range(4)],
        obj_types=[tile_in_ty] * 4,
        names=[f"in1_col1_{i}" for i in range(4)],
        placement=Tile(2, 1)
    )
    in1_col1_cons = of_in1_L2_2_cons

    # in2: Shim 1 -> L2(1,1) -> L2(3,1)
    of_in2_L2_1 = of_in2.cons().forward(tile_in_ty, name="in2_L2_1", placement=Tile(1, 1))
    
    # Split at L2(1,1) into 5: 4 for Col 1, 1 for L2(3,1) (for Col 0)
    # Input stream is [0..7].
    # Chunks 0..3 are for Col 0 (via L2(3,1)).
    # Chunks 4..7 are for Col 1 (local).
    # So we split into [0..3, 4, 5, 6, 7].
    
    of_in2_L2_1_cons = of_in2_L2_1.cons().split(
        [tile_size * 0, tile_size * 4, tile_size * 5, tile_size * 6, tile_size * 7],
        obj_types=[np.ndarray[(4*tile_size,), np.dtype(dtype)], tile_in_ty, tile_in_ty, tile_in_ty, tile_in_ty],
        names=["in2_to_L2_3"] + [f"in2_col1_{i}" for i in range(4)],
        placement=Tile(1, 1)
    )
    
    in2_to_L2_3 = of_in2_L2_1_cons[0]
    in2_col1_cons = of_in2_L2_1_cons[1:5]
    
    # Forward to L2(3,1)
    of_in2_L2_3 = in2_to_L2_3.forward(np.ndarray[(4*tile_size,), np.dtype(dtype)], name="in2_L2_3", placement=Tile(3, 1))
    
    # Split at L2(3,1) to Col 0 tiles
    of_in2_L2_3_cons = of_in2_L2_3.cons().split(
        [tile_size * i for i in range(4)],
        obj_types=[tile_in_ty] * 4,
        names=[f"in2_col0_{i}" for i in range(4)],
        placement=Tile(3, 1)
    )
    in2_col0_cons = of_in2_L2_3_cons

    # out0 producers (join to L2(0,1))
    of_out0_L2_prod = of_out0_L2.prod().join(
        [1 * i for i in range(4)], 
        obj_types=[tile_out_ty] * 4,
        names=[f"out0_prod_{i}" for i in range(4)],
        placement=Tile(0, 1)
    )
    
    # out1 producers (join to L2(1,1))
    of_out1_L2_prod = of_out1_L2.prod().join(
        [1 * i for i in range(4)],
        obj_types=[tile_out_ty] * 4,
        names=[f"out1_prod_{i}" for i in range(4)],
        placement=Tile(1, 1)
    )
    
    of_out0_shim = of_out0_L2.cons().forward(tile_out_ty, name="out0_shim", placement=Tile(0, 0))
    of_out1_shim = of_out1_L2.cons().forward(tile_out_ty, name="out1_shim", placement=Tile(1, 0))

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
            [in1_col0_cons[i].cons(), in2_col0_cons[i].cons(), of_out0_L2_prod[i].prod(), dot_kernel],
            placement=compute_tiles_col0[i]
        ))
        
    # Create workers for Col 1
    for i in range(4):
        workers.append(Worker(
            core_body,
            [in1_col1_cons[i].cons(), in2_col1_cons[i].cons(), of_out1_L2_prod[i].prod(), dot_kernel],
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
        
        # Fill in1 (A) -> L2(0,1)
        rt.fill(of_in1_L2_0.prod(), A, tap_in1, task_group=tg, placement=Tile(0, 0))
        
        # Fill in2 (B) -> L2(1,1)
        rt.fill(of_in2_L2_1.prod(), B, tap_in2, task_group=tg, placement=Tile(1, 0))
        
        # Drain out0 (C0) <- L2(0,1)
        rt.drain(of_out0_shim.cons(), C0, tap_out0, wait=True, task_group=tg, placement=Tile(0, 0))
        
        # Drain out1 (C1) <- L2(1,1)
        rt.drain(of_out1_shim.cons(), C1, tap_out1, wait=True, task_group=tg, placement=Tile(1, 0))
        
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
