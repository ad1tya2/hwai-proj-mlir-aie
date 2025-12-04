#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

import numpy as np
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron.controlflow import range_

def my_dot_product(dev, kernel_size):
    num_columns = 4
    per_tile_elements = kernel_size
    
    # Total elements processed per round (across all columns)
    n = per_tile_elements * num_columns
    
    # For this example, we'll just do 1 round of the specific size
    num_elements = n
    rounds = 1
    
    # Elements per column
    # For BitNet:
    # Input 1 (weights): 2-bit packed -> 1 byte holds 4 elements.
    # Size in bytes = per_tile_elements / 4.
    # Input 2 (activations): 8-bit -> 1 byte holds 1 element.
    # Size in bytes = per_tile_elements.
    
    chunk_in_1 = per_tile_elements // 4
    chunk_in_2 = per_tile_elements
    chunk_out = 1 # 1 output per column per round
    
    # Define tensor types
    # Input 1: uint8
    tensor_in1_ty = np.ndarray[(num_elements // 4,), np.dtype(np.uint8)]
    # Input 2: int8
    tensor_in2_ty = np.ndarray[(num_elements,), np.dtype(np.int8)]
    # Output: float32
    tensor_out_ty = np.ndarray[(num_columns,), np.dtype(np.float32)]
    
    tile_in1_ty = np.ndarray[(chunk_in_1,), np.dtype(np.uint8)]
    tile_in2_ty = np.ndarray[(chunk_in_2,), np.dtype(np.int8)]
    tile_out_ty = np.ndarray[(1,), np.dtype(np.float32)]

    # AIE-array data movement with object fifos
    of_in1s = [ObjectFifo(tile_in1_ty, name=f"in1_{i}") for i in range(num_columns)]
    of_in2s = [ObjectFifo(tile_in2_ty, name=f"in2_{i}") for i in range(num_columns)]
    of_outs = [ObjectFifo(tile_out_ty, name=f"out_{i}") for i in range(num_columns)]

    # Select kernel function based on size
    if kernel_size == 2560:
        kernel_name = "dot_product_i2_i8_2560"
    elif kernel_size == 6912:
        kernel_name = "dot_product_i2_i8_6912"
    else:
        raise ValueError(f"Unsupported kernel size: {kernel_size}")

    # AIE Core Function declaration
    dot_kernel = Kernel(
        kernel_name, "dot.o", [tile_in1_ty, tile_in2_ty, tile_out_ty]
    )

    # Define a task that will run on a compute tile
    def core_body(of_in1, of_in2, of_out, kernel_func):
        # Number of sub-vector "tile" iterations
        for _ in range_(rounds):
            elem_in1 = of_in1.acquire(1)
            elem_in2 = of_in2.acquire(1)
            elem_out = of_out.acquire(1)
            kernel_func(elem_in1, elem_in2, elem_out)
            of_in1.release(1)
            of_in2.release(1)
            of_out.release(1)

    # Create a worker to run the task on a compute tile
    my_workers = [
        Worker(
            core_body,
            [
                of_in1s[i].cons(),
                of_in2s[i].cons(),
                of_outs[i].prod(),
                dot_kernel,
            ],
        )
        for i in range(num_columns)
    ]

    # Create a TensorAccessPattern for each channel
    # Input 1 pattern: distribute chunks to columns (packed weights)
    taps_in1 = [
        TensorAccessPattern(
            (1, num_elements // 4),
            [0, chunk_in_1 * i],
            [1, 1, 1, chunk_in_1],
            [0, 0, 0, 1],
        )
        for i in range(num_columns)
    ]
    
    # Input 2 pattern: distribute chunks to columns (activations)
    taps_in2 = [
        TensorAccessPattern(
            (1, num_elements),
            [0, chunk_in_2 * i],
            [1, 1, 1, chunk_in_2],
            [0, 0, 0, 1],
        )
        for i in range(num_columns)
    ]
    
    # Output pattern: gather results from columns
    taps_out = [
        TensorAccessPattern(
            (1, num_columns),
            [0, 1 * i],
            [1, 1, 1, 1],
            [0, 0, 0, 1],
        )
        for i in range(num_columns)
    ]

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_in1_ty, tensor_in2_ty, tensor_out_ty) as (A, B, C):
        rt.start(*my_workers)

        # Initialize a group for parallel drain tasks
        tg = rt.task_group()

        # Fill the input objectFIFOs with data
        for i in range(num_columns):
            rt.fill(
                of_in1s[i].prod(),
                A,
                taps_in1[i],
                task_group=tg,
            )
            rt.fill(
                of_in2s[i].prod(),
                B,
                taps_in2[i],
                task_group=tg,
            )
        # Drain the output objectFIFOs with data
        for i in range(num_columns):
            rt.drain(
                of_outs[i].cons(),
                C,
                taps_out[i],
                wait=True,
                task_group=tg,
            )
        rt.finish_task_group(tg)

    # Place program components and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", help="Device name", default="npu")
parser.add_argument("-s", "--size", help="Kernel size (2560 or 6912)", type=int, default=2560)
args = parser.parse_args()

if args.device == "npu":
    dev = NPU1()
elif args.device == "npu2":
    dev = NPU2()
else:
    dev = NPU1() # Default

module = my_dot_product(dev, args.size)
print(module)
