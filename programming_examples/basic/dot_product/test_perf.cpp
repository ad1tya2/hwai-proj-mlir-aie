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

using INOUT_DATATYPE_A = uint8_t;
using INOUT_DATATYPE_B = int8_t;
using INOUT_DATATYPE_C = float;

#define QK_I2 128

// --- Reference Code ---

size_t quantize_i2_s_ref(const float * src, void * dst,
                         int64_t n) {
    double max = 0;
    for (int i = 0; i < n; ++i) {
        max = fmax(max, (double) std::fabs((double) src[i]));
    }
    double i2_scale = max;

    uint8_t * q8 = (uint8_t *) std::malloc(n * sizeof(uint8_t));
    if (!q8) return 0;
    for (int i = 0; i < n; ++i) {
        if (std::fabs((double) src[i]) < 1e-6) {
            q8[i] = 1;
            continue;
        }
        q8[i] = (double) src[i] * i2_scale > 0 ? 2 : 0;
    }

    std::memset(dst, 0, n * sizeof(uint8_t) / 4);

    uint8_t * i2_weight = (uint8_t *) dst;
    for (int i = 0; i < n / QK_I2; i++) {
        for (int j = 0; j < QK_I2; j++) {
            int group_idx = j / 32;
            int group_pos = j % 32;
            uint8_t temp = (uint8_t) (q8[i * QK_I2 + j] << (6 - 2 * group_idx));
            i2_weight[i * 32 + group_pos] |= temp;
        }
    }

    // float * scale_ptr = (float *) ((char *) i2_weight + n / 4);
    // scale_ptr[0] = (float) i2_scale;
    // Note: We are not passing scale to AIE in this kernel, just the weights.
    // The AIE kernel just does dot product on the quantized values (0, 1, 2).

    std::free(q8);

    return (size_t) (n / 4);
}

#define QK_I2_S 128

void ggml_vec_dot_i2_i8_s(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    (void) bs;
    (void) bx;
    (void) by;
    (void) nrc;

    // original SIMD kernels ignore offsets and treat pointers as already
    // pointing to the start of the relevant block
    const uint8_t * x = (const uint8_t *) vx;
    const int8_t  * y = (const int8_t  *) vy;

    long long acc = 0;

    // total number of 128-element blocks
    const int nb = n / QK_I2_S; // same as original: nb = n / 128
    (void) nb;

    for (int idx = 0; idx < n; ++idx) {
        // decode the 2-bit q8 for element idx using the same layout as quantize
        const int block = idx / QK_I2;         // which 128-element block
        const int j = idx % QK_I2;             // position inside 128 block (0..127)
        const int group_idx = j / 32;          // 0..3
        const int group_pos = j % 32;          // 0..31
        const size_t byte_index = (size_t) block * 32 + (size_t) group_pos;
        const uint8_t packed = x[byte_index];
        const uint8_t q8 = (packed >> (6 - 2 * group_idx)) & 0x3u; // 2-bit value 0,1,2

        // original SIMD kernels operate directly on 0,1,2 codes without
        // mapping them to -1,0,1 here
        acc += (long long) q8 * (long long) y[idx];
        //normally and without long long see 
    }

    if (s) *s = (float) acc;
}

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
  
  std::cout << "Running test_perf with 4 columns" << std::endl;

  // 4 columns
  int num_columns = 4;
  int INPUT_VOLUME = vector_size; 
  // Output: 1 element per column
  int OUTPUT_VOLUME_PER_COL = 1;

  // Input A: Packed 2-bit weights (uint8)
  size_t INPUT_SIZE_A = (INPUT_VOLUME / 4) * sizeof(INOUT_DATATYPE_A);
  // Input B: Activations (int8)
  size_t INPUT_SIZE_B = INPUT_VOLUME * sizeof(INOUT_DATATYPE_B);
  // Output C: Float
  size_t OUTPUT_SIZE_PER_COL = OUTPUT_VOLUME_PER_COL * sizeof(INOUT_DATATYPE_C);
  
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

  int TOTAL_INPUT_VOLUME = INPUT_VOLUME * num_columns;
  int TOTAL_OUTPUT_VOLUME = num_columns; // 1 per column
  
  size_t TOTAL_INPUT_SIZE_A = (TOTAL_INPUT_VOLUME / 4) * sizeof(INOUT_DATATYPE_A);
  size_t TOTAL_INPUT_SIZE_B = TOTAL_INPUT_VOLUME * sizeof(INOUT_DATATYPE_B);
  size_t TOTAL_OUTPUT_SIZE = TOTAL_OUTPUT_VOLUME * sizeof(INOUT_DATATYPE_C);

  auto bo_inA = xrt::bo(device, TOTAL_INPUT_SIZE_A, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_inB = xrt::bo(device, TOTAL_INPUT_SIZE_B, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_outC = xrt::bo(device, TOTAL_OUTPUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  INOUT_DATATYPE_A *bufInA = bo_inA.map<INOUT_DATATYPE_A *>();
  INOUT_DATATYPE_B *bufInB = bo_inB.map<INOUT_DATATYPE_B *>();
  INOUT_DATATYPE_C *bufOutC = bo_outC.map<INOUT_DATATYPE_C *>();

  // Prepare data
  // Generate random float weights and quantize them
  std::vector<float> weights_float(TOTAL_INPUT_VOLUME);
  for (int i = 0; i < TOTAL_INPUT_VOLUME; i++) weights_float[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  
  quantize_i2_s_ref(weights_float.data(), bufInA, TOTAL_INPUT_VOLUME);

  // Generate random int8 activations
  std::vector<int8_t> activations(TOTAL_INPUT_VOLUME);
  for (int i = 0; i < TOTAL_INPUT_VOLUME; i++) activations[i] = (int8_t)(rand() % 20 - 10);
  memcpy(bufInB, activations.data(), TOTAL_INPUT_SIZE_B);

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
              float total_sum = bufOutC[col];
              float ref_sum = 0;
              
              // Calculate ref sum for this column
              int offset = col * INPUT_VOLUME;
              int offset_packed = col * (INPUT_VOLUME / 4);
              
              ggml_vec_dot_i2_i8_s(INPUT_VOLUME, &ref_sum, 0, bufInA + offset_packed, 0, activations.data() + offset, 0, 0);
              
              if (std::abs(total_sum - ref_sum) > 1e-3) {
                  std::cout << "Verification failed for column " << col << "! Expected " << ref_sum << ", got " << total_sum << std::endl;
                  errors++;
              }
          }
          if (errors > 0) return 1;
          if (verbosity > 0) std::cout << "Verification passed." << std::endl;
      }
  }

  double avg_exec_time = total_exec_time / n_iterations;
  std::cout << "Performance Results (Size " << vector_size << ", 4 Columns):" << std::endl;
  std::cout << "  Average Time: " << avg_exec_time << " us" << std::endl;
  std::cout << "  Min Time:     " << min_exec_time << " us" << std::endl;
  std::cout << "  Max Time:     " << max_exec_time << " us" << std::endl;
  
  double ops = (double)TOTAL_INPUT_VOLUME * 2.0;
  double avg_gops = (ops / (avg_exec_time * 1e-6)) / 1e9;
  std::cout << "  Throughput:   " << avg_gops << " GOPS" << std::endl;
  
  std::cout << "PASS!" << std::endl;
  return 0;
}
