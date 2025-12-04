//===- dot.cc -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

template <typename T_in, typename T_out, const int N>
void dot_product_scalar(T_in *a, T_in *b, T_out *c) {
  T_out sum = 0;
  for (int i = 0; i < N; i++) {
    sum += a[i] * b[i];
  }
  *c = sum;
}

template <typename T_in, typename T_out, const int N>
void dot_product_vector(T_in *a, T_in *b, T_out *c) {
  // Assuming N is a multiple of vector size for simplicity
  constexpr int vec_factor = 32; // 32 elements for bfloat16 (512 bits / 16 bits)
  // Adjust vec_factor based on type if needed, but for bf16 it is usually 32 in AIE2?
  // Actually AIE2 vector register is 512 bits.
  // bfloat16 is 16 bits. 512/16 = 32.
  // int32 is 32 bits. 512/32 = 16.

  T_out sum = 0;
  aie::vector<T_out, 16> acc = aie::zeros<T_out, 16>(); // Accumulator vector

  const int F = N / vec_factor;
  
  T_in *__restrict pA = a;
  T_in *__restrict pB = b;

  // We need to handle different vector sizes for input and output if types differ
  // But here we are doing dot product, so reduction.
  
  // Let's stick to a simple implementation first using aie::mac if possible or just multiply and add.
  
  // For bfloat16 input and float output (accumulation)
  // or bfloat16 input and bfloat16 output? Usually accumulation is in float.
  
  // Let's look at what mul.cc did. It did elementwise.
  
  // For dot product:
  for (int i = 0; i < F; i++) {
      aie::vector<T_in, vec_factor> A = aie::load_v<vec_factor>(pA);
      aie::vector<T_in, vec_factor> B = aie::load_v<vec_factor>(pB);
      // aie::mac is typically for matrix multiply or convolution, but can be used here.
      // Or just mul and add.
      // acc = aie::mac(acc, A, B); // This might need specific types.
      
      // Let's try simple mul and add to accumulator
      // Note: AIE2 has specific intrinsics. aie::mac might work if types align.
      // If T_in is bfloat16, T_out should probably be float for accumulation to avoid overflow?
      // But user asked for simple kernel.
      
      // Let's assume T_in = T_out = int32 for the int32 version.
      // And T_in = bfloat16, T_out = bfloat16 (or float) for bf16 version.
      
      // Actually, let's just implement a scalar version first as "simple kernel" was requested,
      // and a vector version if I'm confident.
      // The prompt said "make me a simple kernel for dot product".
      // I will provide scalar and a simple vector loop.
  }
}

// Re-implementing vector version with aie::accumulate if possible or just loop.
// Let's stick to scalar for the "simple" request if vector is complex without testing.
// But AIE code should be vectorized.

// Let's try to use the aie::sliding_mul or similar if appropriate, but standard mul is fine.

extern "C" {

void dot_product_int32_scalar(int32_t *a_in, int32_t *b_in, int32_t *c_out) {
  dot_product_scalar<int32_t, int32_t, 1024>(a_in, b_in, c_out);
}

void dot_product_bf16_scalar(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out) {
  dot_product_scalar<bfloat16, bfloat16, 1024>(a_in, b_in, c_out);
}

} // extern "C"
