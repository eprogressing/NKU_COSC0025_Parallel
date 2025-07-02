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

// =================================================================
// BARRETT ARITHMETIC HELPERS (HOST)
// =================================================================

// Precompute mu = floor(2^64 / p) for Barrett reduction.
// Requires 128-bit integer support, which nvcc's host compiler (like g++) has.
unsigned long long calculate_mu(unsigned int p) {
    unsigned __int128 mu = 1;
    mu <<= 64;
    return mu / p;
}


// =================================================================
// GPU KERNELS and DEVICE FUNCTIONS (OPTIMIZED WITH BARRETT REDUCTION)
// =================================================================

// Barrett parameters. This struct is passed to kernels.
struct BarrettParams {
    unsigned int p;
    unsigned long long mu; // Precomputed value (2^64 / p)
};

// Barrett Reduction: returns x mod p
// This is the core of the optimization.
// Requirement: x < p^2. This holds for multiplication of two numbers < p.
__device__ unsigned int barrett_reduce(unsigned long long x, const BarrettParams& params) {
    // q = floor((x * mu) / 2^64)
    // This is equivalent to the high 64 bits of the 128-bit product.
    unsigned long long q = ((unsigned __int128)x * params.mu) >> 64;
    // r = x - q * p
    unsigned int r = x - q * params.p;
    // The result is in [0, 2p-1], so one conditional subtraction is needed.
    return (r < params.p) ? r : r - params.p;
}

// Modular multiplication using Barrett reduction.
__device__ unsigned int barrett_mult(unsigned int a, unsigned int b, const BarrettParams& params) {
    unsigned long long prod = (unsigned long long)a * b;
    return barrett_reduce(prod, params);
}

// Modular exponentiation using Barrett reduction.
__device__ unsigned int power_gpu_barrett(unsigned int base, unsigned int exp, const BarrettParams& params) {
    unsigned int res = 1;
    base = barrett_reduce(base, params); // Ensure base is reduced initially
    while (exp > 0) {
        if (exp % 2 == 1) res = barrett_mult(res, base, params);
        base = barrett_mult(base, base, params);
        exp /= 2;
    }
    return res;
}

// Bit reversal kernel (signature changed to unsigned int)
__global__ void bit_reverse_kernel(unsigned int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int logn = 0;
    if (n > 1) logn = __log2f(n);

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

// NTT butterfly stage using Barrett reduction
__global__ void ntt_stage_kernel_barrett(unsigned int* a, int n, int m, bool is_inverse, BarrettParams params) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n / 2) return;

    const int g = 3;
    unsigned int p = params.p;

    unsigned int wm_base = power_gpu_barrett(g, (p - 1) / m, params);
    if (is_inverse) {
        wm_base = power_gpu_barrett(wm_base, p - 2, params);
    }

    int j = tid % (m / 2);
    int k = (tid / (m / 2)) * m;

    int idx1 = k + j;
    int idx2 = idx1 + m / 2;

    unsigned int w = power_gpu_barrett(wm_base, j, params);
    unsigned int t = barrett_mult(w, a[idx2], params);
    unsigned int u = a[idx1];

    // a[idx1] = (u + t) % p
    unsigned int res1 = u + t;
    a[idx1] = (res1 >= p) ? res1 - p : res1;

    // a[idx2] = (u - t + p) % p
    unsigned int res2 = u - t;
    if (u < t) res2 += p;
    a[idx2] = res2;
}

// Pointwise multiplication using Barrett reduction
__global__ void pointwise_mult_kernel_barrett(unsigned int* a, unsigned int* b, unsigned int* ab, int n, BarrettParams params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        ab[i] = barrett_mult(a[i], b[i], params);
    }
}

// Normalization at the end of INTT, using Barrett reduction
__global__ void normalize_kernel_barrett(unsigned int* a, int n, BarrettParams params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned int n_inv = power_gpu_barrett(n, params.p - 2, params);
        a[i] = barrett_mult(a[i], n_inv, params);
    }
}

// =================================================================
// GPU WRAPPER FUNCTION for Barrett NTT
// =================================================================
void ntt_gpu_barrett(unsigned int* d_a, int n, bool is_inverse, BarrettParams params) {
    int threadsPerBlock = 256;

    // Bit reversal
    int blocks_full = (n + threadsPerBlock - 1) / threadsPerBlock;
    bit_reverse_kernel<<<blocks_full, threadsPerBlock>>>(d_a, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Butterfly stages
    for (int m = 2; m <= n; m <<= 1) {
        int stage_blocks = (n / 2 + threadsPerBlock - 1) / threadsPerBlock;
        ntt_stage_kernel_barrett<<<stage_blocks, threadsPerBlock>>>(d_a, n, m, is_inverse, params);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (is_inverse) {
        normalize_kernel_barrett<<<blocks_full, threadsPerBlock>>>(d_a, n, params);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// =================================================================
// MAIN GPU POLYNOMIAL MULTIPLICATION (Optimized)
// =================================================================
void poly_multiply(int* h_a, int* h_b, int* h_ab, int n, int p) {
    int m = 1;
    while (m < 2 * n) { m <<= 1; }

    // --- Host-side Barrett setup ---
    BarrettParams h_params;
    h_params.p = p;
    h_params.mu = calculate_mu(p);

    // --- Prepare data ---
    std::vector<unsigned int> a_padded(m, 0);
    std::vector<unsigned int> b_padded(m, 0);
    for(int i = 0; i < n; i++) {
        a_padded[i] = h_a[i];
        b_padded[i] = h_b[i];
    }

    // --- Allocate GPU Memory ---
    unsigned int *d_a, *d_b;
    size_t size_m = m * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc((void**)&d_a, size_m));
    CUDA_CHECK(cudaMalloc((void**)&d_b, size_m));

    // --- Transfer data to GPU ---
    CUDA_CHECK(cudaMemcpy(d_a, a_padded.data(), size_m, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b_padded.data(), size_m, cudaMemcpyHostToDevice));

    // --- Perform Forward NTTs ---
    ntt_gpu_barrett(d_a, m, false, h_params);
    ntt_gpu_barrett(d_b, m, false, h_params);

    // --- Pointwise Multiplication ---
    int threads = 256;
    int blocks = (m + threads - 1) / threads;
    pointwise_mult_kernel_barrett<<<blocks, threads>>>(d_a, d_b, d_a, m, h_params);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Perform Inverse NTT ---
    ntt_gpu_barrett(d_a, m, true, h_params);

    // --- Copy Final Result to Host ---
    std::vector<unsigned int> ab_padded(m);
    CUDA_CHECK(cudaMemcpy(ab_padded.data(), d_a, size_m, cudaMemcpyDeviceToHost));

    for(int i = 0; i < 2 * n - 1; i++) {
        h_ab[i] = ab_padded[i];
    }

    // --- Free GPU Memory ---
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
}

// =================================================================
// USER'S ORIGINAL FRAMEWORK (UNCHANGED)
// =================================================================
void fRead(int *a, int *b, int *n, int *p, int input_id){
    std::string str1 = "nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strin = str1 + str2 + ".in";

    std::ifstream fin(strin);
    if (!fin.is_open()) {
        std::cerr << "Error opening input file: " << strin << std::endl;
        *n = 0;
        return;
    }
    fin>>*n>>*p;
    for (int i = 0; i < *n; i++){
        fin>>a[i];
    }
    for (int i = 0; i < *n; i++){    
        fin>>b[i];
    }
    fin.close();
}

void fCheck(int *ab, int n, int input_id){
    std::string str1 = "nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";

    std::ifstream fin(strout);
    if (!fin.is_open()) {
        std::cerr << "Error opening output check file: " << strout << std::endl;
        return;
    }
    for (int i = 0; i < n * 2 - 1; i++){
        int x;
        fin>>x;
        if (!fin) {
             std::cout << "Error reading from check file or file ended prematurely." << std::endl;
             break;
        }
        if(x != ab[i]){
            std::cout<<"多项式乘法结果错误 at index " << i << ". Expected " << x << ", got " << ab[i] << std::endl;
            fin.close();
            return;
        }
    }
    std::cout<<"多项式乘法结果正确"<<std::endl;
    fin.close();
}

void fWrite(int *ab, int n, int input_id){
    std::string str1 = "files/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";

    std::ofstream fout(strout);
    for (int i = 0; i < n * 2 - 1; i++){
        fout<<ab[i]<<'\n';
    }
    fout.close();
}

int a[300000], b[300000], ab[600000]; // Increased ab size to avoid overflow

int main(int argc, char *argv[])
{
    int test_begin = 0;
    int test_end = 3; 

    for(int i = test_begin; i <= test_end; ++i){
        long double ans = 0;
        int n_, p_;
        fRead(a, b, &n_, &p_, i);
        if (n_ == 0) continue;

        memset(ab, 0, sizeof(ab));
        auto Start = std::chrono::high_resolution_clock::now();

        poly_multiply(a, b, ab, n_, p_);

        CUDA_CHECK(cudaDeviceSynchronize());
        auto End = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> elapsed = End - Start;
        ans += elapsed.count();

        fCheck(ab, n_, i);
        std::cout<<"Latency for n = "<<n_<<", p = "<<p_<<" : "<<ans<<" (ms) "<<std::endl;

        fWrite(ab, n_, i);
    }
    return 0;
}
