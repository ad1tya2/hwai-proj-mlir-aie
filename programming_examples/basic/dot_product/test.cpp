//===- test.cpp -------------------------------------------------*- C++ -*-===//
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

// Verify results
int verify(int size, std::vector<INOUT_DATATYPE> A, std::vector<INOUT_DATATYPE> B, std::vector<INOUT_DATATYPE> C, int verbosity) {
  int errors = 0;
  int tile_size = 1024;
  int num_outputs = size / tile_size;
  
  if (C.size() < num_outputs) {
      std::cout << "Error: Output size " << C.size() << " is less than expected " << num_outputs << std::endl;
      return 1;
  }

  for (int i = 0; i < num_outputs; i++) {
    INOUT_DATATYPE ref = 0;
    for (int j = 0; j < tile_size; j++) {
        ref += A[i * tile_size + j] * B[i * tile_size + j];
    }
    
    if (C[i] != ref) {
      std::cout << "Error in output " << i << ": " << C[i] << " != " << ref << std::endl;
      errors++;
    } else {
      if (verbosity > 1)
        std::cout << "Correct output " << i << ": " << C[i] << " == " << ref << std::endl;
    }
  }
  return errors;
}

int main(int argc, const char *argv[]) {
  cxxopts::Options options("Dot Product Test");
  cxxopts::ParseResult vm;
  test_utils::add_default_options(options);
  test_utils::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();
  int do_verify = vm["verify"].as<bool>();
  int n_iterations = vm["iters"].as<int>();
  int n_warmup_iterations = vm["warmup"].as<int>();
  int trace_size = vm["trace_sz"].as<int>();

  int INPUT_VOLUME = 65536;
  int OUTPUT_VOLUME = INPUT_VOLUME / 1024;

  size_t INPUT_SIZE = INPUT_VOLUME * sizeof(INOUT_DATATYPE);
  size_t OUTPUT_SIZE = OUTPUT_VOLUME * sizeof(INOUT_DATATYPE);
  
  size_t OUT_BUF_SIZE = OUTPUT_SIZE + trace_size;

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
    std::cerr << "Available kernels:" << std::endl;
    for (auto &k : xkernels) {
      std::cerr << "  " << k.get_name() << std::endl;
    }
    return 1;
  }
  auto xkernel = *xkernel_it;
  auto kernelName = xkernel.get_name();
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_in0 = xrt::bo(device, INPUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_in1 = xrt::bo(device, INPUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out = xrt::bo(device, OUT_BUF_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  INOUT_DATATYPE *bufIn0 = bo_in0.map<INOUT_DATATYPE *>();
  std::vector<INOUT_DATATYPE> AVec(INPUT_VOLUME);
  for (int i = 0; i < INPUT_VOLUME; i++) AVec[i] = rand() % 10;
  memcpy(bufIn0, AVec.data(), INPUT_SIZE);

  INOUT_DATATYPE *bufIn1 = bo_in1.map<INOUT_DATATYPE *>();
  std::vector<INOUT_DATATYPE> BVec(INPUT_VOLUME);
  for (int i = 0; i < INPUT_VOLUME; i++) BVec[i] = rand() % 10;
  memcpy(bufIn1, BVec.data(), INPUT_SIZE);

  INOUT_DATATYPE *bufOut = bo_out.map<INOUT_DATATYPE *>();
  std::vector<INOUT_DATATYPE> CVec(OUTPUT_VOLUME);
  memset(bufOut, 0, OUT_BUF_SIZE);

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned num_iter = n_iterations + n_warmup_iterations;
  
  double min_exec_time = std::numeric_limits<double>::max();
  double max_exec_time = 0;
  double total_exec_time = 0;

  for (unsigned iter = 0; iter < num_iter; iter++) {
      auto start = std::chrono::high_resolution_clock::now();
      auto run = kernel(3, bo_instr, instr_v.size(), bo_in0, bo_in1, bo_out);
      run.wait();
      auto end = std::chrono::high_resolution_clock::now();
      
      if (iter < n_warmup_iterations) continue;
      
      std::chrono::duration<double, std::micro> duration = end - start;
      double exec_time = duration.count();
      
      total_exec_time += exec_time;
      min_exec_time = std::min(min_exec_time, exec_time);
      max_exec_time = std::max(max_exec_time, exec_time);

      bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      memcpy(CVec.data(), bufOut, OUTPUT_SIZE);
      
      if (do_verify) {
          int err = verify(INPUT_VOLUME, AVec, BVec, CVec, verbosity);
          if (err) {
              std::cout << "Verification failed with " << err << " errors." << std::endl;
              return 1;
          } else {
              if (verbosity > 0) std::cout << "Verification passed." << std::endl;
          }
      }
  }

  double avg_exec_time = total_exec_time / n_iterations;
  std::cout << "Performance Results:" << std::endl;
  std::cout << "  Average Time: " << avg_exec_time << " us" << std::endl;
  std::cout << "  Min Time:     " << min_exec_time << " us" << std::endl;
  std::cout << "  Max Time:     " << max_exec_time << " us" << std::endl;
  
  // Calculate MACs: 65536 elements * 1 MAC/element = 65536 MACs per run
  // GFLOPS = (MACs * 2) / (Time in seconds * 1e9)  (Assuming 2 ops per MAC: mul + add)
  // Or just GOPS.
  double ops = (double)INPUT_VOLUME * 2.0; // 2 ops per element (mul + add)
  double avg_gops = (ops / (avg_exec_time * 1e-6)) / 1e9;
  std::cout << "  Throughput:   " << avg_gops << " GOPS" << std::endl;
  
  std::cout << "PASS!" << std::endl;
  return 0;
}
