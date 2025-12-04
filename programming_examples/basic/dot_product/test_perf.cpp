//===- test_perf.cpp --------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 AMD Inc.

#include <bits/stdc++.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "cxxopts.hpp"
#include "test_utils.h"

using INOUT_DATATYPE = int32_t;

int main(int argc, const char *argv[]) {
  cxxopts::Options options("Dot Product Perf Test");
  cxxopts::ParseResult vm;
  test_utils::add_default_options(options);
  options.add_options()("size", "vector size", cxxopts::value<int>()->default_value("2560"));
  
  test_utils::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();
  int do_verify = vm["verify"].as<bool>();
  int n_iterations = vm["iters"].as<int>();
  int n_warmup_iterations = vm["warmup"].as<int>();
  int trace_size = vm["trace_sz"].as<int>();
  int vector_size = vm["size"].as<int>();

  // 5 columns
  int num_columns = 5;
  int INPUT_VOLUME = vector_size; 
  // Output: 1 element per column
  int OUTPUT_VOLUME_PER_COL = 1;

  size_t INPUT_SIZE = INPUT_VOLUME * sizeof(INOUT_DATATYPE);
  size_t OUTPUT_SIZE_PER_COL = OUTPUT_VOLUME_PER_COL * sizeof(INOUT_DATATYPE);
  
  srand(time(NULL));

  std::vector<uint32_t> instr_v = test_utils::load_instr_binary(vm["instr"].as<std::string>());

  unsigned int device_index = 0;
  auto device = xrt::device(device_index);
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());
  std::string Node = vm["kernel"].as<std::string>();
  
  auto xkernels = xclbin.get_kernels();
  auto xkernel_it = std::find_if(xkernels.begin(), xkernels.end(),
                               [Node](xrt::xclbin::kernel &k) {
                                 return k.get_name().rfind(Node, 0) == 0;
                               });
  if (xkernel_it == xkernels.end()) {
    std::cerr << "Error: Kernel '" << Node << "' not found in xclbin." << std::endl;
    return 1;
  }
  auto xkernel = *xkernel_it;
  auto kernelName = xkernel.get_name();
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  // Buffers: 5 input pairs (A, B) and 5 outputs (C)
  // Arguments start from index 3.
  // in1_0, in2_0, out_0, in1_1, in2_1, out_1, ...
  // Wait, let's check dot.py argument order.
  // my_workers = [Worker(..., [of_in1s[i], of_in2s[i], of_outs[i], ...])]
  // rt.sequence(tensor_in_ty, tensor_in_ty, tensor_out_ty)
  // The runtime sequence defines the external buffers.
  // rt.sequence(A, B, C)
  // A is input 1 (all columns share A? No, A is one large tensor).
  // In dot.py:
  // tensor_in_ty = np.ndarray[(num_elements,), ...]
  // taps_in = [TensorAccessPattern(..., chunk_in * i, ...)]
  // So there are only 3 external buffers: A, B, C.
  // A and B are split by the shim DMA.
  // C is gathered by the shim DMA.
  
  // So we only need 3 BOs.
  // Wait, let's verify dot.py again.
  // with rt.sequence(tensor_in_ty, tensor_in_ty, tensor_out_ty) as (A, B, C):
  // Yes, 3 buffers.
  
  int TOTAL_INPUT_VOLUME = INPUT_VOLUME * num_columns;
  int TOTAL_OUTPUT_VOLUME = num_columns; // 1 per column
  
  size_t TOTAL_INPUT_SIZE = TOTAL_INPUT_VOLUME * sizeof(INOUT_DATATYPE);
  size_t TOTAL_OUTPUT_SIZE = TOTAL_OUTPUT_VOLUME * sizeof(INOUT_DATATYPE);

  auto bo_inA = xrt::bo(device, TOTAL_INPUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_inB = xrt::bo(device, TOTAL_INPUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_outC = xrt::bo(device, TOTAL_OUTPUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  INOUT_DATATYPE *bufInA = bo_inA.map<INOUT_DATATYPE *>();
  std::vector<INOUT_DATATYPE> AVec(TOTAL_INPUT_VOLUME);
  for (int i = 0; i < TOTAL_INPUT_VOLUME; i++) AVec[i] = rand() % 10;
  memcpy(bufInA, AVec.data(), TOTAL_INPUT_SIZE);

  INOUT_DATATYPE *bufInB = bo_inB.map<INOUT_DATATYPE *>();
  std::vector<INOUT_DATATYPE> BVec(TOTAL_INPUT_VOLUME);
  for (int i = 0; i < TOTAL_INPUT_VOLUME; i++) BVec[i] = rand() % 10;
  memcpy(bufInB, BVec.data(), TOTAL_INPUT_SIZE);

  INOUT_DATATYPE *bufOutC = bo_outC.map<INOUT_DATATYPE *>();
  memset(bufOutC, 0, TOTAL_OUTPUT_SIZE);

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_outC.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned num_iter = n_iterations + n_warmup_iterations;
  
  double min_exec_time = std::numeric_limits<double>::max();
  double max_exec_time = 0;
  double total_exec_time = 0;

  for (unsigned iter = 0; iter < num_iter; iter++) {
      auto start = std::chrono::high_resolution_clock::now();
      auto run = kernel(3, bo_instr, instr_v.size(), bo_inA, bo_inB, bo_outC);
      run.wait();
      auto end = std::chrono::high_resolution_clock::now();
      
      if (iter < n_warmup_iterations) continue;
      
      std::chrono::duration<double, std::micro> duration = end - start;
      double exec_time = duration.count();
      
      total_exec_time += exec_time;
      min_exec_time = std::min(min_exec_time, exec_time);
      max_exec_time = std::max(max_exec_time, exec_time);

      bo_outC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      
      if (do_verify) {
          int errors = 0;
          for(int col=0; col<num_columns; col++) {
              INOUT_DATATYPE total_sum = bufOutC[col];
              INOUT_DATATYPE ref_sum = 0;
              // Calculate ref sum for this column
              // Data is interleaved or blocked?
              // taps_in: chunk_in * i. Blocked.
              int offset = col * INPUT_VOLUME;
              for(int i=0; i<INPUT_VOLUME; i++) {
                  ref_sum += AVec[offset + i] * BVec[offset + i];
              }
              
              if (total_sum != ref_sum) {
                  std::cout << "Verification failed for column " << col << "! Expected " << ref_sum << ", got " << total_sum << std::endl;
                  errors++;
              }
          }
          if (errors > 0) return 1;
          if (verbosity > 0) std::cout << "Verification passed." << std::endl;
      }
  }

  double avg_exec_time = total_exec_time / n_iterations;
  std::cout << "Performance Results (Size " << vector_size << ", 5 Columns):" << std::endl;
  std::cout << "  Average Time: " << avg_exec_time << " us" << std::endl;
  std::cout << "  Min Time:     " << min_exec_time << " us" << std::endl;
  std::cout << "  Max Time:     " << max_exec_time << " us" << std::endl;
  
  double ops = (double)TOTAL_INPUT_VOLUME * 2.0;
  double avg_gops = (ops / (avg_exec_time * 1e-6)) / 1e9;
  std::cout << "  Throughput:   " << avg_gops << " GOPS" << std::endl;
  
  std::cout << "PASS!" << std::endl;
  return 0;
}
