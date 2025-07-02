#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>

// CUDA 错误检查宏
#define CUDA_CHECK(err) { \
    cudaError_t e = err; \
    if (e != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(e) << " in " << __FILE__ \
                  << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Barrett Parameters
struct BarrettParams {
    unsigned int p;
    unsigned long long mu;
};

// Host-side Barrett mu calculation
unsigned long long calculate_mu(unsigned int p) {
    unsigned __int128 mu = 1;
    mu <<= 64;
    return mu / p;
}

// Device Barrett Reduction
__device__ unsigned int barrett_reduce(unsigned long long x, const BarrettParams& params) {
    unsigned long long q = ((unsigned __int128)x * params.mu) >> 64;
    unsigned int r = x - q * params.p;
    return (r < params.p) ? r : r - params.p;
}

// Device Barrett Multiplication
__device__ unsigned int barrett_mult(unsigned int a, unsigned int b, const BarrettParams& params) {
    unsigned long long prod = (unsigned long long)a * b;
    return barrett_reduce(prod, params);
}

// Device Modular Exponentiation
__device__ unsigned int power_gpu_barrett(unsigned int base, unsigned int exp, const BarrettParams& params) {
    unsigned int res = 1;
    base = barrett_reduce(base, params);
    while (exp > 0) {
        if (exp & 1) res = barrett_mult(res, base, params);
        base = barrett_mult(base, base, params);
        exp >>= 1;
    }
    return res;
}

// Bit Reversal Kernel
__global__ void bit_reverse_kernel(unsigned int* a, int n, int logn) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int rev_i = 0;
    for (int j = 0; j < logn; j++) {
        if ((i >> j) & 1) {
            rev_i |= 1 << (logn - 1 - j);
        }
    }

    if (i < rev_i) {
        unsigned int temp = a[i];
        a[i] = a[rev_i];
        a[rev_i] = temp;
    }
}

// Optimized NTT Kernel with Shared Memory
__global__ void ntt_stage_kernel_shared(unsigned int* a, unsigned int* w_precomputed, int n, int stage, int logn, bool is_inverse, BarrettParams params) {
    extern __shared__ unsigned int shmem[];
    int tid = threadIdx.x;
    int block_start = blockIdx.x * blockDim.x * 2;
    if (block_start >= n) return;

    int m = 1 << stage;
    int half_m = m >> 1;
    int global_idx1 = block_start + tid;
    int global_idx2 = global_idx1 + half_m;

    // Load data into shared memory
    if (global_idx1 < n) shmem[tid] = a[global_idx1];
    if (global_idx2 < n) shmem[tid + half_m] = a[global_idx2];
    __syncthreads();

    if (tid < half_m && global_idx1 < n) {
        int w_idx = (blockIdx.x * half_m + tid) + (is_inverse ? (logn - stage - 1) : stage) * (n / 2);
        unsigned int w = w_precomputed[w_idx];
        unsigned int u = shmem[tid];
        unsigned int t = barrett_mult(w, shmem[tid + half_m], params);

        unsigned int res1 = u + t;
        shmem[tid] = (res1 >= params.p) ? res1 - params.p : res1;

        unsigned int res2 = u - t;
        shmem[tid + half_m] = (u < t) ? res2 + params.p : res2;
    }
    __syncthreads();

    // Write back to global memory
    if (global_idx1 < n) a[global_idx1] = shmem[tid];
    if (global_idx2 < n) a[global_idx2] = shmem[tid + half_m];
}

// Pointwise Multiplication Kernel
__global__ void pointwise_mult_kernel_barrett(unsigned int* a, unsigned int* b, unsigned int* ab, int n, BarrettParams params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        ab[i] = barrett_mult(a[i], b[i], params);
    }
}

// Normalization Kernel
__global__ void normalize_kernel_barrett(unsigned int* a, int n, unsigned int n_inv, BarrettParams params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] = barrett_mult(a[i], n_inv, params);
    }
}

// Precompute Butterfly Factors
void precompute_butterfly_factors(std::vector<unsigned int>& w_precomputed, int n, bool is_inverse, BarrettParams params) {
    const int g = 3;
    int logn = log2(n);
    w_precomputed.resize(n * logn / 2);
    int idx = 0;

    for (int stage = 0; stage < logn; stage++) {
        int m = 1 << (stage + 1);
        int half_m = m >> 1;
        unsigned int wm_base = power_gpu_barrett(g, (params.p - 1) / m, params);
        if (is_inverse) wm_base = power_gpu_barrett(wm_base, params.p - 2, params);

        for (int j = 0; j < half_m; j++) {
            w_precomputed[idx++] = power_gpu_barrett(wm_base, j, params);
        }
    }
}

// NTT GPU Wrapper
void ntt_gpu_barrett_opt(unsigned int* d_a, int n, bool is_inverse, BarrettParams params, unsigned int* d_w_precomputed, cudaStream_t stream) {
    int threadsPerBlock = 256;
    int logn = log2(n);

    // Bit reversal
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    bit_reverse_kernel<<<blocks, threadsPerBlock, 0, stream>>>(d_a, n, logn);
    CUDA_CHECK(cudaGetLastError());

    // NTT stages with shared memory
    for (int stage = 0; stage < logn; stage++) {
        int m = 1 << (stage + 1);
        int half_m = m >> 1;
        int num_blocks = (n / m + threadsPerBlock - 1) / threadsPerBlock;
        size_t shared_size = threadsPerBlock * 2 * sizeof(unsigned int);
        ntt_stage_kernel_shared<<<num_blocks, threadsPerBlock, shared_size, stream>>>(d_a, d_w_precomputed, n, stage, logn, is_inverse, params);
        CUDA_CHECK(cudaGetLastError());
    }

    if (is_inverse) {
        unsigned int n_inv = power_gpu_barrett(n, params.p - 2, params);
        normalize_kernel_barrett<<<blocks, threadsPerBlock, 0, stream>>>(d_a, n, n_inv, params);
        CUDA_CHECK(cudaGetLastError());
    }
}

// Polynomial Multiplication
void poly_multiply(int* h_a, int* h_b, int* h_ab, int n, int p) {
    int m = 1;
    while (m < 2 * n) m <<= 1;

    // Barrett setup
    BarrettParams h_params;
    h_params.p = p;
    h_params.mu = calculate_mu(p);

    // Prepare padded data
    std::vector<unsigned int> a_padded(m, 0), b_padded(m, 0);
    for (int i = 0; i < n; i++) {
        a_padded[i] = h_a[i];
        b_padded[i] = h_b[i];
    }

    // Allocate GPU memory
    unsigned int *d_a, *d_b, *d_w_forward, *d_w_inverse;
    size_t size_m = m * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(&d_a, size_m));
    CUDA_CHECK(cudaMalloc(&d_b, size_m));
    CUDA_CHECK(cudaMalloc(&d_w_forward, m * log2(m) / 2 * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_w_inverse, m * log2(m) / 2 * sizeof(unsigned int)));

    // Precompute butterfly factors
    std::vector<unsigned int> w_forward, w_inverse;
    precompute_butterfly_factors(w_forward, m, false, h_params);
    precompute_butterfly_factors(w_inverse, m, true, h_params);

    // Create streams
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    // Asynchronous data transfer
    CUDA_CHECK(cudaMemcpyAsync(d_a, a_padded.data(), size_m, cudaMemcpyHostToDevice, stream1));
    CUDA_CHECK(cudaMemcpyAsync(d_b, b_padded.data(), size_m, cudaMemcpyHostToDevice, stream2));
    CUDA_CHECK(cudaMemcpyAsync(d_w_forward, w_forward.data(), w_forward.size() * sizeof(unsigned int), cudaMemcpyHostToDevice, stream1));
    CUDA_CHECK(cudaMemcpyAsync(d_w_inverse, w_inverse.data(), w_inverse.size() * sizeof(unsigned int), cudaMemcpyHostToDevice, stream2));

    // Forward NTTs
    ntt_gpu_barrett_opt(d_a, m, false, h_params, d_w_forward, stream1);
    ntt_gpu_barrett_opt(d_b, m, false, h_params, d_w_forward, stream2);

    // Pointwise multiplication
    int threads = 256;
    int blocks = (m + threads - 1) / threads;
    pointwise_mult_kernel_barrett<<<blocks, threads, 0, stream1>>>(d_a, d_b, d_a, m, h_params);
    CUDA_CHECK(cudaGetLastError());

    // Inverse NTT
    ntt_gpu_barrett_opt(d_a, m, true, h_params, d_w_inverse, stream1);

    // Copy result back
    std::vector<unsigned int> ab_padded(m);
    CUDA_CHECK(cudaMemcpyAsync(ab_padded.data(), d_a, size_m, cudaMemcpyDeviceToHost, stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream1));

    for (int i = 0; i < 2 * n - 1; i++) {
        h_ab[i] = ab_padded[i];
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_w_forward));
    CUDA_CHECK(cudaFree(d_w_inverse));
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
}

// File I/O Functions (Unchanged)
void fRead(int *a, int *b, int *n, int *p, int input_id) {
    std::string str1 = "nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strin = str1 + str2 + ".in";

    std::ifstream fin(strin);
    if (!fin.is_open()) {
        std::cerr << "Error opening input file: " << strin << std::endl;
        *n = 0;
        return;
    }
    fin >> *n >> *p;
    for (int i = 0; i < *n; i++) fin >> a[i];
    for (int i = 0; i < *n; i++) fin >> b[i];
    fin.close();
}

void fCheck(int *ab, int n, int input_id) {
    std::string str1 = "nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";

    std::ifstream fin(strout);
    if (!fin.is_open()) {
        std::cerr << "Error opening output check file: " << strout << std::endl;
        return;
    }
    for (int i = 0; i < n * 2 - 1; i++) {
        int x;
        fin >> x;
        if (!fin) {
            std::cout << "Error reading from check file or file ended prematurely." << std::endl;
            break;
        }
        if (x != ab[i]) {
            std::cout << "多项式乘法结果错误 at index " << i << ". Expected " << x << ", got " << ab[i] << std::endl;
            fin.close();
            return;
        }
    }
    std::cout << "多项式乘法结果正确" << std::endl;
    fin.close();
}

void fWrite(int *ab, int n, int input_id) {
    std::string str1 = "files/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";

    std::ofstream fout(strout);
    for (int i = 0; i < n * 2 - 1; i++) {
        fout << ab[i] << '\n';
    }
    fout.close();
}

int a[300000], b[300000], ab[600000];

int main(int argc, char *argv[]) {
    int test_begin = 0;
    int test_end = 3;

    for (int i = test_begin; i <= test_end; ++i) {
        long double ans = 0;
        int n_, p_;
        fRead(a, b, &n_, &p_, i);
        if (n_ == 0) continue;

        memset(ab, 0, sizeof(ab));
        auto Start = std::chrono::high_resolution_clock::now();

        poly_multiply(a, b, ab, n_, p_);

        CUDA_CHECK(cudaDeviceSynchronize());
        auto End = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = End - Start;
        ans += elapsed.count();

        fCheck(ab, n_, i);
        std::cout << "Latency for n = " << n_ << ", p = " << p_ << " : " << ans << " (ms) " << std::endl;

        fWrite(ab, n_, i);
    }
    return 0;
}
