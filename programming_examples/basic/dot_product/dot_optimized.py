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
    # We define the logical graph and let the library handle placement.
    
    num_tiles = 8
    
    if num_elements % num_tiles != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be divisible by {num_tiles}."
        )
        
    chunk_size = num_elements // num_tiles
    tile_size = chunk_size
    
    dtype = np.int32

    # Define tensor types
    tensor_in_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    # Output: 1 partial sum per tile -> 8 partial sums total
    # We will collect 4 on Shim 0 and 4 on Shim 1 (logically)
    tensor_out_ty = np.ndarray[(4,), np.dtype[dtype]] 
    
    tile_in_ty = np.ndarray[(tile_size,), np.dtype[dtype]]
    tile_out_ty = np.ndarray[(1,), np.dtype[dtype]]

    # --- Object FIFOs & Routing ---
    # We use separate FIFOs for inputs and outputs to avoid channel exhaustion on a single tile.
    # We use a daisy-chain topology for inputs: Shim -> L2_0 -> L2_1 -> Workers
    
    # --- Input 1 (in1) ---
    # in1 L2 primary (feeds workers 0-3 and forwards to secondary)
    of_in1_L2_0 = ObjectFifo(tile_in_ty, name="in1_L2_0")
    
    # Split L2_0: 4 chunks for local workers, 1 chunk for forwarding
    of_in1_L2_0_cons = of_in1_L2_0.cons().split(
        [tile_size * 0, tile_size * 1, tile_size * 2, tile_size * 3, tile_size * 4],
        obj_types=[tile_in_ty, tile_in_ty, tile_in_ty, tile_in_ty, np.ndarray[(4*tile_size,), np.dtype(dtype)]],
        names=[f"in1_w{i}" for i in range(4)] + ["in1_fwd"]
    )
    in1_w0_3 = of_in1_L2_0_cons[0:4]
    in1_fwd = of_in1_L2_0_cons[4]
    
    # Forward to secondary L2
    of_in1_L2_1 = in1_fwd.cons().forward(obj_type=np.ndarray[(4*tile_size,), np.dtype(dtype)], name="in1_L2_1")
    
    # Split L2_1: 4 chunks for workers 4-7
    in1_w4_7 = of_in1_L2_1.cons().split(
        [tile_size * i for i in range(4)],
        obj_types=[tile_in_ty] * 4,
        names=[f"in1_w{i}" for i in range(4, 8)]
    )

    # --- Input 2 (in2) ---
    # in2 L2 primary (feeds workers 0-3 and forwards to secondary)
    of_in2_L2_0 = ObjectFifo(tile_in_ty, name="in2_L2_0")
    
    # Split L2_0: 4 chunks for local workers, 1 chunk for forwarding
    of_in2_L2_0_cons = of_in2_L2_0.cons().split(
        [tile_size * 0, tile_size * 1, tile_size * 2, tile_size * 3, tile_size * 4],
        obj_types=[tile_in_ty, tile_in_ty, tile_in_ty, tile_in_ty, np.ndarray[(4*tile_size,), np.dtype(dtype)]],
        names=[f"in2_w{i}" for i in range(4)] + ["in2_fwd"]
    )
    in2_w0_3 = of_in2_L2_0_cons[0:4]
    in2_fwd = of_in2_L2_0_cons[4]
    
    # Forward to secondary L2
    of_in2_L2_1 = in2_fwd.cons().forward(obj_type=np.ndarray[(4*tile_size,), np.dtype(dtype)], name="in2_L2_1")
    
    # Split L2_1: 4 chunks for workers 4-7
    in2_w4_7 = of_in2_L2_1.cons().split(
        [tile_size * i for i in range(4)],
        obj_types=[tile_in_ty] * 4,
        names=[f"in2_w{i}" for i in range(4, 8)]
    )

    # --- Outputs ---
    
    # out0 L2 (collects from workers 0-3)
    of_out0_L2 = ObjectFifo(tile_out_ty, name="out0_L2")
    
    # out0 producers (join to L2 from 4 workers)
    of_out0_L2_prod = of_out0_L2.prod().join(
        [1 * i for i in range(4)], 
        obj_types=[tile_out_ty] * 4,
        names=[f"out0_prod_{i}" for i in range(4)]
    )
    
    # out1 L2 (collects from workers 4-7)
    of_out1_L2 = ObjectFifo(tile_out_ty, name="out1_L2")
    
    # out1 producers (join to L2 from 4 workers)
    of_out1_L2_prod = of_out1_L2.prod().join(
        [1 * i for i in range(4)],
        obj_types=[tile_out_ty] * 4,
        names=[f"out1_prod_{i}" for i in range(4)]
    )

    # --- Kernel ---
    dot_kernel = Kernel(
        "dot_product_int32_scalar", "dot.o", [tile_in_ty, tile_in_ty, tile_out_ty]
    )

    # --- Workers ---
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
    # Create workers for Group 0 (fed by in1_L2_0, in2_L2_0, out to out0)
    for i in range(4):
        workers.append(Worker(
            core_body,
            [in1_w0_3[i].cons(), in2_w0_3[i].cons(), of_out0_L2_prod[i].prod(), dot_kernel]
        ))
        
    # Create workers for Group 1 (fed by in1_L2_1, in2_L2_1, out to out1)
    for i in range(4):
        workers.append(Worker(
            core_body,
            [in1_w4_7[i].cons(), in2_w4_7[i].cons(), of_out1_L2_prod[i].prod(), dot_kernel]
        ))

    # --- Runtime ---
    # TAPs
    tap_in1 = TensorAccessPattern((1, num_elements), 0, [1, 1, 1, num_elements], [0, 0, 0, 1])
    tap_in2 = TensorAccessPattern((1, num_elements), 0, [1, 1, 1, num_elements], [0, 0, 0, 1])
    tap_out0 = TensorAccessPattern((1, 4), 0, [1, 1, 1, 4], [0, 0, 0, 1])
    tap_out1 = TensorAccessPattern((1, 4), 0, [1, 1, 1, 4], [0, 0, 0, 1])

    rt = Runtime()
    with rt.sequence(tensor_in_ty, tensor_in_ty, tensor_out_ty, tensor_out_ty) as (A, B, C0, C1):
        rt.start(*workers)
        tg = rt.task_group()
        
        # Fill in1 -> L2_0
        rt.fill(of_in1_L2_0.prod(), A, tap_in1, task_group=tg)
        
        # Fill in2 -> L2_0
        rt.fill(of_in2_L2_0.prod(), B, tap_in2, task_group=tg)
        
        # Drain out0 <- L2
        rt.drain(of_out0_L2.cons(), C0, tap_out0, wait=True, task_group=tg)
        
        # Drain out1 <- L2
        rt.drain(of_out1_L2.cons(), C1, tap_out1, wait=True, task_group=tg)
        
        # Explicitly register intermediate FIFOs with Runtime so they are seen by the Placer
        # This is necessary because they are not used in fill/drain
        rt._fifos.add(of_in1_L2_1.prod())
        rt._fifos.add(of_in2_L2_1.prod())
        
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
