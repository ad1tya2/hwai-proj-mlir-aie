// ggml-bitnet-mad-scalar.cpp
// Scalar implementation that preserves the original packing / decoding logic.

#include <vector>
#include <type_traits>

#include "ggml-bitnet.h"
#include "ggml-quants.h"
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
// for size_t and malloc/memset when compiled as C++
#include <cstddef>
#include <iostream>
#include <set>
#include <dlfcn.h>
#include <mutex>
using namespace std;
#define QK_I2_S 128
#define QK_I2 128

// AIE Library Handles
static void* libdot_handle = nullptr;
typedef int (*init_kernels_t)(const char*, const char*);
typedef void (*run_2560_t)(uint8_t*, int8_t*, float*);
typedef void (*run_6912_t)(uint8_t*, int8_t*, float*);

static init_kernels_t init_kernels_ptr = nullptr;
static run_2560_t run_2560_ptr = nullptr;
static run_6912_t run_6912_ptr = nullptr;
static std::mutex aie_mutex;
static bool aie_initialized = false;
static bool aie_load_failed = false;

void load_aie_lib() {
    std::lock_guard<std::mutex> lock(aie_mutex);
    if (aie_initialized || aie_load_failed) return;

    // Try to load from current directory or build directory
    const char* lib_paths[] = {"./libdot.so", "./build/libdot.so", "../libdot.so"};
    for (const char* path : lib_paths) {
        libdot_handle = dlopen(path, RTLD_LAZY);
        if (libdot_handle) break;
    }

    if (!libdot_handle) {
        std::cerr << "Failed to load libdot.so: " << dlerror() << std::endl;
        aie_load_failed = true;
        return;
    }

    init_kernels_ptr = (init_kernels_t)dlsym(libdot_handle, "init_kernels");
    run_2560_ptr = (run_2560_t)dlsym(libdot_handle, "run_2560");
    run_6912_ptr = (run_6912_t)dlsym(libdot_handle, "run_6912");

    if (!init_kernels_ptr || !run_2560_ptr || !run_6912_ptr) {
        std::cerr << "Failed to load symbols from libdot.so" << std::endl;
        dlclose(libdot_handle);
        libdot_handle = nullptr;
        aie_load_failed = true;
        return;
    }

    // Initialize kernels
    // Assuming xclbin is in build/
    if (init_kernels_ptr("build/dot_lib.xclbin", "build/dot_lib_insts.bin") != 0) {
        std::cerr << "Failed to initialize AIE kernels" << std::endl;
        dlclose(libdot_handle);
        libdot_handle = nullptr;
        aie_load_failed = true;
        return;
    }

    aie_initialized = true;
    std::cout << "AIE Library Loaded and Initialized Successfully" << std::endl;
}

std::set<int> seen;

size_t quantize_i2_s(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void) quant_weights; // unused in this routine, kept for API compatibility

    // total number of elements
    const int64_t n = nrow * n_per_row;

    // compute global max abs (same as original)
    double max_abs = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double v = fabs((double) src[i]);
        if (v > max_abs) max_abs = v;
    }
    double i2_scale = max_abs;

    // build q8 array (values 0,1,2 as in original)
    uint8_t * q8 = (uint8_t *) malloc((size_t) n * sizeof(uint8_t));
    if (!q8) return 0;
    for (int64_t i = 0; i < n; ++i) {
        if (fabs((double) src[i]) < 1e-6) {
            q8[i] = 1;
            continue;
        }
        // original: q8[i] = (double)src[i] * i2_scale > 0 ? 2 : 0;
        q8[i] = (double) src[i] * i2_scale > 0 ? 2 : 0;
    }

    // zero output buffer for packed bytes: total packed bytes = n / 4
    // (since 4 values per byte overall when packing 2 bits each)
    const size_t total_packed_bytes = (size_t) (n / 4); // original used this
    std::memset(dst, 0, total_packed_bytes);

    uint8_t * i2_weight = (uint8_t *) dst;

    // Pack following original loops
    const int64_t n_blocks = n / QK_I2; // number of 128-element blocks
    for (int64_t block = 0; block < n_blocks; ++block) {
        const int64_t base = block * QK_I2;
        for (int j = 0; j < QK_I2; ++j) {
            int group_idx = j / 32;       // 0..3
            int group_pos = j % 32;       // 0..31
            uint8_t temp = (uint8_t) (q8[base + j] << (6 - 2 * group_idx));
            i2_weight[block * 32 + group_pos] |= temp;
        }
    }

    // write the global scale float at offset (char*)i2_weight + n/4 (same as original)
    float * scale_ptr = (float *) ((char *) i2_weight + total_packed_bytes);
    scale_ptr[0] = (float) i2_scale;

    free(q8);

    // match original return: nrow * row_size / 4 + 32 (32B for alignment)
    size_t row_size = ggml_row_size(GGML_TYPE_I2_S, n_per_row);
    return nrow * row_size / 4 + 32;
}

// Scalar decoder/dot product that follows the packing above.
// Decodes q8 for element index idx from x layout and multiplies with y[idx].
void ggml_vec_dot_i2_i8_s(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    (void) bs;
    (void) bx;
    (void) by;
    (void) nrc;

    if (seen.insert(n).second) {  // true if n was not already in the set
        std::cout << "New n: " << n << '\n';
    }

    // Try to load library if not loaded
    if (!aie_initialized && !aie_load_failed) {
        load_aie_lib();
    }

    if (aie_initialized) {
        std::lock_guard<std::mutex> lock(aie_mutex);
        if (n == 2560) {
            run_2560_ptr((uint8_t*)vx, (int8_t*)vy, s);
            return;
        } else if (n == 6912) {
            run_6912_ptr((uint8_t*)vx, (int8_t*)vy, s);
            return;
        }
    }
    
    // Fallback removed as requested.
    std::cerr << "Error: AIE kernel not run for n=" << n << " (Initialized: " << aie_initialized << ")" << std::endl;
    if (s) *s = 0.0f;
}
