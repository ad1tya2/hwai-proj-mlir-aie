//===- test_optimized.cpp ---------------------------------------*- C++ -*-===//
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
  cxxopts::Options options("Dot Product Optimized Test");
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

  int INPUT_VOLUME = vector_size;
  // Output: 4 elements from Col 0, 4 elements from Col 1
  int OUTPUT_VOLUME_0 = 4;
  int OUTPUT_VOLUME_1 = 4;

  size_t INPUT_SIZE = INPUT_VOLUME * sizeof(INOUT_DATATYPE);
  size_t OUTPUT_SIZE_0 = OUTPUT_VOLUME_0 * sizeof(INOUT_DATATYPE);
  size_t OUTPUT_SIZE_1 = OUTPUT_VOLUME_1 * sizeof(INOUT_DATATYPE);
  
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

  // Buffers: in1, in2, out0, out1
  auto bo_in1 = xrt::bo(device, INPUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_in2 = xrt::bo(device, INPUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out0 = xrt::bo(device, OUTPUT_SIZE_0, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  auto bo_out1 = xrt::bo(device, OUTPUT_SIZE_1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));

  INOUT_DATATYPE *bufIn1 = bo_in1.map<INOUT_DATATYPE *>();
  std::vector<INOUT_DATATYPE> AVec(INPUT_VOLUME);
  for (int i = 0; i < INPUT_VOLUME; i++) AVec[i] = rand() % 10;
  memcpy(bufIn1, AVec.data(), INPUT_SIZE);

  INOUT_DATATYPE *bufIn2 = bo_in2.map<INOUT_DATATYPE *>();
  std::vector<INOUT_DATATYPE> BVec(INPUT_VOLUME);
  for (int i = 0; i < INPUT_VOLUME; i++) BVec[i] = rand() % 10;
  memcpy(bufIn2, BVec.data(), INPUT_SIZE);

  INOUT_DATATYPE *bufOut0 = bo_out0.map<INOUT_DATATYPE *>();
  INOUT_DATATYPE *bufOut1 = bo_out1.map<INOUT_DATATYPE *>();
  
  memset(bufOut0, 0, OUTPUT_SIZE_0);
  memset(bufOut1, 0, OUTPUT_SIZE_1);

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out1.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned num_iter = n_iterations + n_warmup_iterations;
  
  double min_exec_time = std::numeric_limits<double>::max();
  double max_exec_time = 0;
  double total_exec_time = 0;

  for (unsigned iter = 0; iter < num_iter; iter++) {
      auto start = std::chrono::high_resolution_clock::now();
      auto run = kernel(3, bo_instr, instr_v.size(), bo_in1, bo_in2, bo_out0, bo_out1);
      run.wait();
      auto end = std::chrono::high_resolution_clock::now();
      
      if (iter < n_warmup_iterations) continue;
      
      std::chrono::duration<double, std::micro> duration = end - start;
      double exec_time = duration.count();
      
      total_exec_time += exec_time;
      min_exec_time = std::min(min_exec_time, exec_time);
      max_exec_time = std::max(max_exec_time, exec_time);

      bo_out0.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      bo_out1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      
      if (do_verify) {
          INOUT_DATATYPE total_sum = 0;
          for(int i=0; i<4; i++) total_sum += bufOut0[i];
          for(int i=0; i<4; i++) total_sum += bufOut1[i];
          
          INOUT_DATATYPE ref_sum = 0;
          for(int i=0; i<INPUT_VOLUME; i++) ref_sum += AVec[i] * BVec[i];
          
          if (total_sum != ref_sum) {
              std::cout << "Verification failed! Expected " << ref_sum << ", got " << total_sum << std::endl;
              return 1;
          } else {
              if (verbosity > 0) std::cout << "Verification passed." << std::endl;
          }
      }
  }

  double avg_exec_time = total_exec_time / n_iterations;
  std::cout << "Performance Results (Size " << vector_size << "):" << std::endl;
  std::cout << "  Average Time: " << avg_exec_time << " us" << std::endl;
  std::cout << "  Min Time:     " << min_exec_time << " us" << std::endl;
  std::cout << "  Max Time:     " << max_exec_time << " us" << std::endl;
  
  double ops = (double)INPUT_VOLUME * 2.0;
  double avg_gops = (ops / (avg_exec_time * 1e-6)) / 1e9;
  std::cout << "  Throughput:   " << avg_gops << " GOPS" << std::endl;
  
  std::cout << "PASS!" << std::endl;
  return 0;
}
