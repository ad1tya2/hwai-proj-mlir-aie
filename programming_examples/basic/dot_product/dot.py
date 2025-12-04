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

def my_dot_product(dev, num_elements, trace_size):
    num_columns = 4
    # If the device is NPU2, we use 4 columns for this example (or 8 if available, but let's stick to 4)
    
    per_tile_elements = 1024
    # Total elements processed per round (across all columns)
    n = per_tile_elements * num_columns
    
    if num_elements % n != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {n}."
        )
    
    # Number of rounds
    rounds = num_elements // n
    
    # Elements per column
    chunk_in = num_elements // num_columns
    chunk_out = rounds # 1 output per round per column
    
    dtype = np.int32

    # Define tensor types
    tensor_in_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    # Output has one result per 1024 elements
    tensor_out_ty = np.ndarray[(num_elements // per_tile_elements,), np.dtype[dtype]]
    
    tile_in_ty = np.ndarray[(per_tile_elements,), np.dtype[dtype]]
    tile_out_ty = np.ndarray[(1,), np.dtype[dtype]]

    # AIE-array data movement with object fifos
    of_in1s = [ObjectFifo(tile_in_ty, name=f"in1_{i}") for i in range(num_columns)]
    of_in2s = [ObjectFifo(tile_in_ty, name=f"in2_{i}") for i in range(num_columns)]
    of_outs = [ObjectFifo(tile_out_ty, name=f"out_{i}") for i in range(num_columns)]

    # AIE Core Function declaration
    # dot_product_int32_scalar takes (int32*, int32*, int32*)
    dot_kernel = Kernel(
        "dot_product_int32_scalar", "dot.o", [tile_in_ty, tile_in_ty, tile_out_ty]
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
    # Input pattern: distribute chunks to columns
    taps_in = [
        TensorAccessPattern(
            (1, num_elements),
            chunk_in * i,
            [1, 1, 1, chunk_in],
            [0, 0, 0, 1],
        )
        for i in range(num_columns)
    ]
    
    # Output pattern: gather results from columns
    # Total output elements = num_elements / 1024
    # Each column produces chunk_out elements
    taps_out = [
        TensorAccessPattern(
            (1, num_elements // per_tile_elements),
            chunk_out * i,
            [1, 1, 1, chunk_out],
            [0, 0, 0, 1],
        )
        for i in range(num_columns)
    ]

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_in_ty, tensor_in_ty, tensor_out_ty) as (A, B, C):
        rt.start(*my_workers)

        # Initialize a group for parallel drain tasks
        tg = rt.task_group()

        # Fill the input objectFIFOs with data
        for i in range(num_columns):
            rt.fill(
                of_in1s[i].prod(),
                A,
                taps_in[i],
                task_group=tg,
            )
            rt.fill(
                of_in2s[i].prod(),
                B,
                taps_in[i],
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


try:
    device_name = str(sys.argv[1])
    if device_name == "npu":
        dev = NPU1()
    elif device_name == "npu2":
        dev = NPU2()
    else:
        # Default to NPU1 if not specified or unknown, or raise error
        # The makefile passes -d ${DEVICE} which is npu or npu2
        # But sys.argv[1] is the device name passed by makefile
        pass
        
    # The makefile calls: python3 $< -d ${DEVICE} > $@
    # Wait, the makefile says: python3 $< -d ${DEVICE} > $@
    # But sys.argv parsing in eltwise_mul.py was:
    # device_name = str(sys.argv[1])
    # This implies arguments are passed positionally?
    # Let's check Makefile again.
    # Makefile: python3 $< -d ${DEVICE} > $@
    # This looks like flags.
    # But eltwise_mul.py did: device_name = str(sys.argv[1])
    # If I run `python3 eltwise_mul.py -d npu`, sys.argv[1] is "-d".
    # So eltwise_mul.py parsing might be wrong or Makefile is different.
    # In eltwise_mul/Makefile:
    # ${mlir_aie_mul}: ${srcdir}/${eltwise_mul_py}
    # 	python3 $< ${DEVICE} > $@
    # Ah, I should check the Makefile I wrote.
    # I wrote: python3 $< -d ${DEVICE} > $@
    # I should change my Makefile to match eltwise_mul.py's expectation OR change python script to parse flags.
    # I'll change python script to use argparse or just handle the flag.
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", help="Device name", default="npu")
    args = parser.parse_args()
    
    if args.device == "npu":
        dev = NPU1()
    elif args.device == "npu2":
        dev = NPU2()
    else:
        dev = NPU1() # Default

    trace_size = 0 
    # Hardcoded num_elements for generation, but runtime can handle dynamic?
    # Usually we generate for a specific size or max size.
    # eltwise_mul.py generated for 65536.
    # 65536 / 1024 = 64 tiles.
    # 4 columns -> 16 tiles per column.
    module = my_dot_product(dev, 65536, trace_size)
    print(module)

except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
