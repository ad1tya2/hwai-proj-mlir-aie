//===- dot_lib.cpp ------------------------------------------------*- C++ -*-===//
//
// Shared library for AIE BitNet Dot Product
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <vector>
#include <cstring>
#include <stdexcept>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// Configuration
#define MAX_SIZE 6912
#define SIZE_2560 2560
#define SIZE_6912 6912

// Global state
struct AIEContext {
    xrt::device device;
    xrt::kernel kernel;
    xrt::bo bo_instr;
    xrt::bo bo_inA;
    xrt::bo bo_inB;
    xrt::bo bo_outC;
    void* bufInstr;
    uint8_t* bufInA;
    int8_t* bufInB;
    float* bufOutC;
    bool initialized = false;
};

static AIEContext ctx;

extern "C" {

// Initialize the AIE kernel
// xclbin_path: Path to the .xclbin file
// instr_path: Path to the .bin instruction file
int init_kernels(const char* xclbin_path, const char* instr_path) {
    if (ctx.initialized) return 0;

    try {
        // Load instructions
        std::vector<uint32_t> instr_v;
        FILE* f = fopen(instr_path, "rb");
        if (!f) {
            std::cerr << "Error: Could not open instruction file " << instr_path << std::endl;
            return -1;
        }
        fseek(f, 0, SEEK_END);
        long fsize = ftell(f);
        fseek(f, 0, SEEK_SET);
        instr_v.resize(fsize / sizeof(uint32_t));
        fread(instr_v.data(), sizeof(uint32_t), instr_v.size(), f);
        fclose(f);

        // Open Device
        ctx.device = xrt::device(0); // Index 0
        auto xclbin = xrt::xclbin(xclbin_path);
        
        // Register XCLBIN
        ctx.device.register_xclbin(xclbin);
        xrt::hw_context context(ctx.device, xclbin.get_uuid());
        
        // Get Kernel
        // Assuming kernel name is "dot_product_i2_i8_6912" as we use the max size
        // Or "MLIR_AIE" as generic name? In dot.py we use "dot_product_i2_i8_6912" for the function,
        // but the XRT kernel name is usually "MLIR_AIE" (the top level).
        ctx.kernel = xrt::kernel(context, "MLIR_AIE");

        // Allocate Buffers (Max Size)
        // Input A: Packed 2-bit weights (uint8) -> MAX_SIZE / 4
        size_t size_A = (MAX_SIZE / 4) * sizeof(uint8_t);
        // Input B: Activations (int8) -> MAX_SIZE
        size_t size_B = MAX_SIZE * sizeof(int8_t);
        // Output C: Float -> 1 element (single tile, single column)
        size_t size_C = 1 * sizeof(float);
        
        // Instruction Buffer
        ctx.bo_instr = xrt::bo(ctx.device, instr_v.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, ctx.kernel.group_id(1));
        ctx.bufInstr = ctx.bo_instr.map<void*>();
        memcpy(ctx.bufInstr, instr_v.data(), instr_v.size() * sizeof(int));
        ctx.bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // Data Buffers
        ctx.bo_inA = xrt::bo(ctx.device, size_A, XRT_BO_FLAGS_HOST_ONLY, ctx.kernel.group_id(3));
        ctx.bo_inB = xrt::bo(ctx.device, size_B, XRT_BO_FLAGS_HOST_ONLY, ctx.kernel.group_id(4));
        ctx.bo_outC = xrt::bo(ctx.device, size_C, XRT_BO_FLAGS_HOST_ONLY, ctx.kernel.group_id(5));

        ctx.bufInA = ctx.bo_inA.map<uint8_t*>();
        ctx.bufInB = ctx.bo_inB.map<int8_t*>();
        ctx.bufOutC = ctx.bo_outC.map<float*>();

        ctx.initialized = true;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Init Error: " << e.what() << std::endl;
        return -1;
    }
}

// Run 2560 size kernel
// weights: Packed 2-bit weights (2560/4 = 640 bytes)
// activations: int8 activations (2560 bytes)
// result: float output (1 element)
void run_2560(uint8_t* weights, int8_t* activations, float* result) {
    if (!ctx.initialized) return;

    // Copy data
    memcpy(ctx.bufInA, weights, (SIZE_2560 / 4));
    memcpy(ctx.bufInB, activations, SIZE_2560);

    // Pad the rest with zeros
    memset(ctx.bufInA + (SIZE_2560 / 4), 0, (MAX_SIZE - SIZE_2560) / 4);
    memset(ctx.bufInB + SIZE_2560, 0, MAX_SIZE - SIZE_2560);

    // Sync to device
    ctx.bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    ctx.bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Run
    auto run = ctx.kernel(3, ctx.bo_instr, ctx.bo_instr.size() / sizeof(int), ctx.bo_inA, ctx.bo_inB, ctx.bo_outC);
    run.wait();

    // Sync back
    ctx.bo_outC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    *result = ctx.bufOutC[0];
}

// Run 6912 size kernel
void run_6912(uint8_t* weights, int8_t* activations, float* result) {
    if (!ctx.initialized) return;

    // Copy data (Full size)
    memcpy(ctx.bufInA, weights, (SIZE_6912 / 4));
    memcpy(ctx.bufInB, activations, SIZE_6912);

    // Sync to device
    ctx.bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    ctx.bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Run
    auto run = ctx.kernel(3, ctx.bo_instr, ctx.bo_instr.size() / sizeof(int), ctx.bo_inA, ctx.bo_inB, ctx.bo_outC);
    run.wait();

    // Sync back
    ctx.bo_outC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    *result = ctx.bufOutC[0];
}

void close_kernels() {
    // XRT objects destructors handle cleanup
    ctx.initialized = false;
}

} // extern "C"
