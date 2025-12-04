#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <dlfcn.h>
#include <mutex>
#include <set>

using namespace std;

#define QK_I2_S 128
#define QK_I2   128

// --- AIE Library Loading Logic (from ggml-mad.cpp) ---

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

// --- Simple scalar version (Reference for verification) ---

size_t quantize_i2_s_simple(const float * src, void * dst,
                            int64_t nrow, int64_t n_per_row,
                            const float * /*quant_weights*/) {
    const int64_t n = nrow * n_per_row;

    double max_abs = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double v = std::fabs((double) src[i]);
        if (v > max_abs) max_abs = v;
    }
    double i2_scale = max_abs;

    uint8_t * q8 = (uint8_t *) std::malloc((size_t) n * sizeof(uint8_t));
    if (!q8) return 0;
    for (int64_t i = 0; i < n; ++i) {
        if (std::fabs((double) src[i]) < 1e-6) {
            q8[i] = 1;
            continue;
        }
        q8[i] = (double) src[i] * i2_scale > 0 ? 2 : 0;
    }

    const size_t total_packed_bytes = (size_t) (n / 4);
    std::memset(dst, 0, total_packed_bytes);

    uint8_t * i2_weight = (uint8_t *) dst;

    const int64_t n_blocks = n / QK_I2;
    for (int64_t block = 0; block < n_blocks; ++block) {
        const int64_t base = block * QK_I2;
        for (int j = 0; j < QK_I2; ++j) {
            int group_idx = j / 32;
            int group_pos = j % 32;
            uint8_t temp = (uint8_t) (q8[base + j] << (6 - 2 * group_idx));
            i2_weight[block * 32 + group_pos] |= temp;
        }
    }

    float * scale_ptr = (float *) ((char *) i2_weight + total_packed_bytes);
    scale_ptr[0] = (float) i2_scale;

    std::free(q8);

    // row_size not important here; just return bytes used
    return total_packed_bytes + sizeof(float);
}

void ggml_vec_dot_i2_i8_s_simple(int n, float * s,
                                 const void * vx, const void * vy) {
    const uint8_t * x = (const uint8_t *) vx;
    const int8_t  * y = (const int8_t  *) vy;

    long long acc = 0;

    for (int idx = 0; idx < n; ++idx) {
        const int block = idx / QK_I2;
        const int j = idx % QK_I2;
        const int group_idx = j / 32;
        const int group_pos = j % 32;
        const size_t byte_index = (size_t) block * 32 + (size_t) group_pos;
        const uint8_t packed = x[byte_index];
        const uint8_t q8 = (packed >> (6 - 2 * group_idx)) & 0x3u;

        // simple version currently uses raw 0/1/2 codes (after our change)
        acc += (long long) q8 * (long long) y[idx];
    }

    if (s) *s = (float) acc;
}

// --- AIE Library Wrapper (Replaces original ref) ---

void ggml_vec_dot_i2_i8_s(int n, float * s, const void * vx,  const void * vy) {
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
    
    std::cerr << "Error: AIE kernel not run for n=" << n << " (Initialized: " << aie_initialized << ")" << std::endl;
    if (s) *s = 0.0f;
}

// --- Test harness comparing Simple vs AIE Lib ---

int main() {
    const int64_t nrow = 1;
    // Use 2560 to test AIE library support
    const int64_t n_per_row = 2560; 
    const int n = (int) (nrow * n_per_row);

    std::cout << "Testing with n=" << n << std::endl;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> src(n);
    for (int i = 0; i < n; ++i) src[i] = dist(rng);

    std::vector<uint8_t> buf_simple(n / 4 + sizeof(float) + 64);
    // Use same buffer for both since quantization logic is same (simple)
    // Actually, let's use simple quantization for both inputs to ensure consistency
    
    size_t used_simple = quantize_i2_s_simple(src.data(), buf_simple.data(),
                                              nrow, n_per_row, nullptr);

    std::printf("bytes simple = %zu\n", used_simple);

    std::vector<int8_t> y(n);
    std::uniform_int_distribution<int> dist_i8(-10, 10);
    for (int i = 0; i < n; ++i) y[i] = (int8_t) dist_i8(rng);

    float s_simple = 0.0f;
    float s_aie    = 0.0f;

    // Run Simple Scalar
    ggml_vec_dot_i2_i8_s_simple(n, &s_simple, buf_simple.data(), y.data());
    
    // Run AIE Library
    ggml_vec_dot_i2_i8_s(n, &s_aie, buf_simple.data(), y.data());

    std::printf("dot_simple = %f\n", s_simple);
    std::printf("dot_aie    = %f\n", s_aie);

    const float diff = std::fabs(s_simple - s_aie);
    std::printf("abs diff   = %f\n", diff);

    if (diff > 1e-3f) {
        std::printf("MISMATCH larger than tolerance!\n");
        return 1;
    }

    std::printf("OK: simple and AIE match within tolerance.\n");
    return 0;
}