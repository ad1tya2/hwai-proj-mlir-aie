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

// Optimized kernel for size 2560
// 2560 elements. Vector size for int32 is 16 (512/32).
// 2560 / 16 = 160 iterations.
// Unroll by 4 -> 40 iterations.
void dot_product_2560(int32_t *a_in, int32_t *b_in, int32_t *c_out) {
  aie::accum<acc64, 16> acc = aie::zeros<acc64, 16>();
  int32_t *__restrict pA = a_in;
  int32_t *__restrict pB = b_in;

  for (int i = 0; i < 40; i++) chess_prepare_for_pipelining {
    aie::vector<int32_t, 16> A0 = aie::load_v<16>(pA); pA += 16;
    aie::vector<int32_t, 16> B0 = aie::load_v<16>(pB); pB += 16;
    acc = aie::mac(acc, A0, B0);

    aie::vector<int32_t, 16> A1 = aie::load_v<16>(pA); pA += 16;
    aie::vector<int32_t, 16> B1 = aie::load_v<16>(pB); pB += 16;
    acc = aie::mac(acc, A1, B1);

    aie::vector<int32_t, 16> A2 = aie::load_v<16>(pA); pA += 16;
    aie::vector<int32_t, 16> B2 = aie::load_v<16>(pB); pB += 16;
    acc = aie::mac(acc, A2, B2);

    aie::vector<int32_t, 16> A3 = aie::load_v<16>(pA); pA += 16;
    aie::vector<int32_t, 16> B3 = aie::load_v<16>(pB); pB += 16;
    acc = aie::mac(acc, A3, B3);
  }

  *c_out = aie::reduce_add(acc.to_vector<int32_t>(0));
}

// Optimized kernel for size 6912
// 6912 elements. Vector size for int32 is 16.
// 6912 / 16 = 432 iterations.
// Unroll by 4 -> 108 iterations.
void dot_product_6912(int32_t *a_in, int32_t *b_in, int32_t *c_out) {
  aie::accum<acc64, 16> acc = aie::zeros<acc64, 16>();
  int32_t *__restrict pA = a_in;
  int32_t *__restrict pB = b_in;

  for (int i = 0; i < 108; i++) chess_prepare_for_pipelining {
    aie::vector<int32_t, 16> A0 = aie::load_v<16>(pA); pA += 16;
    aie::vector<int32_t, 16> B0 = aie::load_v<16>(pB); pB += 16;
    acc = aie::mac(acc, A0, B0);

    aie::vector<int32_t, 16> A1 = aie::load_v<16>(pA); pA += 16;
    aie::vector<int32_t, 16> B1 = aie::load_v<16>(pB); pB += 16;
    acc = aie::mac(acc, A1, B1);

    aie::vector<int32_t, 16> A2 = aie::load_v<16>(pA); pA += 16;
    aie::vector<int32_t, 16> B2 = aie::load_v<16>(pB); pB += 16;
    acc = aie::mac(acc, A2, B2);

    aie::vector<int32_t, 16> A3 = aie::load_v<16>(pA); pA += 16;
    aie::vector<int32_t, 16> B3 = aie::load_v<16>(pB); pB += 16;
    acc = aie::mac(acc, A3, B3);
  }

  *c_out = aie::reduce_add(acc.to_vector<int32_t>(0));
}

// BitNet 2-bit quantized dot product
// n: number of elements (must be multiple of 128)
// vx: packed 2-bit weights (uint8_t*)
// vy: int8_t activations (int8_t*)
// s: output float scalar
// Note: In this kernel signature we use standard pointers.
// The caller passes pointers to buffers.
// We accumulate to float at the end.
void dot_product_i2_i8(int32_t n, uint8_t *vx, int8_t *vy, float *s) {
  // Accumulator for the result
  // We use acc32 for int8*int8 accumulation.
  // Max value of int8 is 127. Max value of weight is 2.
  // 127 * 2 = 254.
  // 2560 elements * 254 = 650240. Fits in int32.
  // Even for 6912, it fits.
  aie::accum<acc32, 32> acc = aie::zeros<acc32, 32>();
  
  uint8_t *__restrict px = vx;
  int8_t *__restrict py = vy;

  // Process 128 elements per iteration
  // 128 elements = 32 bytes of packed weights
  // 128 elements = 128 bytes of activations
  int blocks = n / 128;
  for (int i = 0; i < blocks; i++) chess_prepare_for_pipelining {
    // Load 32 bytes of packed weights
    aie::vector<uint8_t, 32> v_packed = aie::load_v<32>(px);
    px += 32;

    // Unpack weights
    // v0: indices 0..31 (bits 6-7)
    // v1: indices 32..63 (bits 4-5)
    // v2: indices 64..95 (bits 2-3)
    // v3: indices 96..127 (bits 0-1)
    
    // We need to cast to int8 for multiplication with int8 activations
    // But first unpack as uint8 to handle shifts correctly
    
    // Mask 0x3 = 00000011
    aie::vector<uint8_t, 32> mask = aie::broadcast<uint8_t, 32>(0x3);
    
    // Right shift by multiplication is tricky for integers in AIE without specific intrinsics.
    // However, AIE2 has vshift instructions.
    // Let's try to use standard C++ operators on the vector if possible, but the error said no.
    // The error said "invalid operands to binary expression ('aie::vector<uint8_t, 32>' ... and 'int')".
    // This means operator>> is not overloaded for vector and int.
    
    // We can use aie::utils::bit_slice if available, or just manual masking and shifting if we can cast to something that supports it.
    // Or we can use `aie::srl` (Shift Right Logical) if it exists.
    // Let's try `aie::srl`.
    // If `aie::srl` is not found, we might need to look deeper.
    // But wait, `upshift` was lane shift.
    
    // Let's try to use `aie::shuffle_down`? No.
    
    // Actually, for 2-bit unpacking, we can use a look-up table (LUT) approach or just simple masking if we can shift.
    // Since we can't shift easily, maybe we can use `aie::mul` with a fractional type? No, these are uint8.
    
    // Let's try to cast to `aie::accum` which might support shifting?
    // Or use `aie::vector_cast`.
    
    // Wait, I can use `aie::sbs` (Select Bits) or similar?
    
    // Let's try `aie::srl` which is standard in some AIE APIs.
    // aie::vector<uint8_t, 32> w0_u = aie::bit_and(aie::srl(v_packed, 6), mask);
    
    // If aie::srl doesn't exist, I will try to implement it via multiplication if possible, but uint8 multiplication is limited.
    
    // Another option: The user mentioned "make it simpler and avoid the thousands of errors you can handle dimensions in main kernel if required".
    // Maybe I can just use a loop?
    // "for (int k=0; k<32; k++) ..."
    // The compiler might vectorize it.
    
    aie::vector<uint8_t, 32> w0_u;
    aie::vector<uint8_t, 32> w1_u;
    aie::vector<uint8_t, 32> w2_u;
    aie::vector<uint8_t, 32> w3_u;

    // w3_u = bit_and(v_packed, mask);
    // w2_u = bit_and(logical_downshift(v_packed, 2), mask);
    // w1_u = bit_and(logical_downshift(v_packed, 4), mask);
    // w0_u = bit_and(logical_downshift(v_packed, 6), mask);

    for(int k=0; k<32; k++) {
        uint8_t val = v_packed[k];
        w0_u[k] = (val >> 6) & 0x3;
        w1_u[k] = (val >> 4) & 0x3;
        w2_u[k] = (val >> 2) & 0x3;
        w3_u[k] = (val >> 0) & 0x3;
    }
        

    // Cast to int8
    aie::vector<int8_t, 32> w0 = aie::vector_cast<int8_t>(w0_u);
    aie::vector<int8_t, 32> w1 = aie::vector_cast<int8_t>(w1_u);
    aie::vector<int8_t, 32> w2 = aie::vector_cast<int8_t>(w2_u);
    aie::vector<int8_t, 32> w3 = aie::vector_cast<int8_t>(w3_u);

    // Load activations
    aie::vector<int8_t, 32> y0 = aie::load_v<32>(py); py += 32;
    aie::vector<int8_t, 32> y1 = aie::load_v<32>(py); py += 32;


    // Multiply and accumulate
    acc = aie::mac(acc, w0, y0);
    acc = aie::mac(acc, w1, y1);
    aie::vector<int8_t, 32> y2 = aie::load_v<32>(py); py += 32;
    aie::vector<int8_t, 32> y3 = aie::load_v<32>(py); py += 32;
    acc = aie::mac(acc, w2, y2);
    acc = aie::mac(acc, w3, y3);
  }

  int32_t sum = aie::reduce_add(acc.to_vector<int32_t>(0));
  *s = (float)sum;
}

void dot_product_i2_i8_2560(uint8_t *vx, int8_t *vy, float *s) {
    dot_product_i2_i8(2560, vx, vy, s);
}

void dot_product_i2_i8_6912(uint8_t *vx, int8_t *vy, float *s) {
    dot_product_i2_i8(6912, vx, vy, s);
}

} // extern "C"
