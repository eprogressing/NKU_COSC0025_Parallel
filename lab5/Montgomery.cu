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
// MONTGOMERY ARITHMETIC HELPERS (HOST)
// =================================================================

// Calculate p_prime for Montgomery reduction. p_prime = -p^-1 mod R, where R = 2^32.
// This uses the Newton-Raphson method for finding modular inverses modulo powers of 2.
unsigned int calculate_p_prime(unsigned int p) {
    if (p % 2 == 0) {
        std::cerr << "Montgomery reduction requires an odd modulus." << std::endl;
        exit(EXIT_FAILURE);
    }
    unsigned int p_inv = 1;
    for (int i = 0; i < 4; ++i) { // 2 -> 4 -> 8 -> 16 -> 32 bits of precision
        p_inv = p_inv * (2 - p * p_inv);
    }
    return -p_inv;
}

// Calculate R^2 mod p, where R = 2^32.
unsigned int calculate_r2_mod_p(unsigned int p) {
    // Using __int128 for simplicity, as it's supported by nvcc's frontend (GCC/Clang).
    unsigned __int128 r_squared = 1;
    r_squared <<= 64; // R^2 for R=2^32
    return r_squared % p;
}


// =================================================================
// GPU KERNELS and DEVICE FUNCTIONS (OPTIMIZED WITH MONTGOMERY)
// =================================================================

// Montgomery parameters. This struct is passed to kernels.
struct MontParams {
    unsigned int p;       // Modulus
    unsigned int p_prime; // -p^-1 mod R (R=2^32)
    unsigned int r2_mod_p; // R^2 mod p, where R=2^32
};

// Montgomery Reduction: returns (t * R^-1) mod p
// This is the core of the optimization, replacing division with shifts and muls.
// Requirement: t < p*R
__device__ unsigned int mont_reduce(unsigned long long t, const MontParams& params) {
    // m = (t * p') mod R
    unsigned int m = (unsigned int)t * params.p_prime;
    // t = (t + m*p) / R
    unsigned int res = (unsigned int)((t + (unsigned long long)m * params.p) >> 32);
    // Conditional subtraction to ensure result is in [0, p-1]
    return (res < params.p) ? res : res - params.p;
}

// Montgomery Multiplication: returns (a_mont * b_mont * R^-1) mod p
// Inputs a_mont and b_mont are assumed to be in Montgomery form.
__device__ unsigned int mont_mult(unsigned int a_mont, unsigned int b_mont, const MontParams& params) {
    unsigned long long prod = (unsigned long long)a_mont * b_mont;
    return mont_reduce(prod, params);
}

// Convert a number to Montgomery form: returns (a * R) mod p
__device__ unsigned int to_mont_form(unsigned int a, const MontParams& params) {
    // We multiply by R^2 and then reduce. (a * R^2) * R^-1 mod p = (a * R) mod p
    return mont_mult(a, params.r2_mod_p, params);
}

// Convert a number from Montgomery form: returns (a_mont * 1) mod p
__device__ unsigned int from_mont_form(unsigned int a_mont, const MontParams& params) {
    // (a_mont * R^-1) mod p
    return mont_reduce(a_mont, params);
}

// Modular exponentiation using Montgomery multiplication.
// Base is in standard form, result is in standard form.
__device__ unsigned int mont_power(unsigned int base, unsigned int exp, const MontParams& params) {
    unsigned int res_mont = to_mont_form(1, params);
    unsigned int base_mont = to_mont_form(base, params);
    while (exp > 0) {
        if (exp % 2 == 1) res_mont = mont_mult(res_mont, base_mont, params);
        base_mont = mont_mult(base_mont, base_mont, params);
        exp /= 2;
    }
    return from_mont_form(res_mont, params);
}

// Kernel to convert an entire array to Montgomery form
__global__ void to_mont_form_kernel(unsigned int* d_out, const unsigned int* d_in, int n, MontParams params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        d_out[i] = to_mont_form(d_in[i], params);
    }
}

// Kernel to convert an entire array back from Montgomery form
__global__ void from_mont_form_kernel(unsigned int* d_out, const unsigned int* d_in, int n, MontParams params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        d_out[i] = from_mont_form(d_in[i], params);
    }
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

// NTT butterfly stage using Montgomery multiplication
__global__ void ntt_stage_kernel_mont(unsigned int* a, int n, int m, bool is_inverse, MontParams params) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n / 2) return;

    const int g = 3;
    unsigned int p = params.p;

    // Calculate base twiddle factor w_m. Result is in standard form.
    unsigned int wm_base_std = mont_power(g, (p - 1) / m, params);
    if (is_inverse) {
        wm_base_std = mont_power(wm_base_std, p - 2, params);
    }

    int j = tid % (m / 2);
    int k = (tid / (m / 2)) * m;

    int idx1 = k + j;
    int idx2 = idx1 + m / 2;

    // Calculate w = (w_m)^j. This is slow but matches original logic.
    // The main optimization comes from mont_mult inside the power function.
    unsigned int w_std = mont_power(wm_base_std, j, params);
    unsigned int w_mont = to_mont_form(w_std, params);

    // Butterfly operation on data in Montgomery form
    unsigned int u = a[idx1];
    unsigned int v = a[idx2];
    unsigned int t = mont_mult(w_mont, v, params);

    // (u + t) mod p. Addition is the same in Montgomery domain.
    unsigned int res1 = u + t;
    a[idx1] = (res1 >= p) ? res1 - p : res1;

    // (u - t) mod p. Subtraction is also the same.
    unsigned int res2 = u - t;
    if (u < t) res2 += p; // Handle potential underflow
    a[idx2] = res2;
}

// Pointwise multiplication in Montgomery domain
__global__ void pointwise_mult_kernel_mont(unsigned int* a, unsigned int* b, unsigned int* ab, int n, MontParams params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        ab[i] = mont_mult(a[i], b[i], params);
    }
}

// Normalization at the end of INTT, using Montgomery multiplication
__global__ void normalize_kernel_mont(unsigned int* a, int n, MontParams params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Calculate n_inv in standard form, then convert for multiplication
        unsigned int n_inv_std = mont_power(n, params.p - 2, params);
        unsigned int n_inv_mont = to_mont_form(n_inv_std, params);
        a[i] = mont_mult(a[i], n_inv_mont, params);
    }
}

// =================================================================
// GPU WRAPPER FUNCTION for Montgomery NTT
// =================================================================
void ntt_gpu_mont(unsigned int* d_a, int n, bool is_inverse, MontParams params) {
    int threadsPerBlock = 256;

    // Bit reversal
    int blocks_full = (n + threadsPerBlock - 1) / threadsPerBlock;
    bit_reverse_kernel<<<blocks_full, threadsPerBlock>>>(d_a, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Butterfly stages
    for (int m = 2; m <= n; m <<= 1) {
        int stage_blocks = (n / 2 + threadsPerBlock - 1) / threadsPerBlock;
        ntt_stage_kernel_mont<<<stage_blocks, threadsPerBlock>>>(d_a, n, m, is_inverse, params);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (is_inverse) {
        normalize_kernel_mont<<<blocks_full, threadsPerBlock>>>(d_a, n, params);
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

    // --- Host-side Montgomery setup ---
    unsigned int p_uint = p;
    MontParams h_params;
    h_params.p = p_uint;
    h_params.p_prime = calculate_p_prime(p_uint);
    h_params.r2_mod_p = calculate_r2_mod_p(p_uint);

    // --- Prepare data ---
    // Use unsigned int for calculations
    std::vector<unsigned int> a_padded(m, 0);
    std::vector<unsigned int> b_padded(m, 0);
    for(int i = 0; i < n; i++) {
        a_padded[i] = h_a[i];
        b_padded[i] = h_b[i];
    }

    // --- Allocate GPU Memory ---
    unsigned int *d_a_std, *d_b_std, *d_a_mont, *d_b_mont;
    size_t size_m = m * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc((void**)&d_a_std, size_m));
    CUDA_CHECK(cudaMalloc((void**)&d_b_std, size_m));
    CUDA_CHECK(cudaMalloc((void**)&d_a_mont, size_m));
    CUDA_CHECK(cudaMalloc((void**)&d_b_mont, size_m));

    // --- Transfer and Convert to Montgomery Form ---
    CUDA_CHECK(cudaMemcpy(d_a_std, a_padded.data(), size_m, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_std, b_padded.data(), size_m, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (m + threads - 1) / threads;
    to_mont_form_kernel<<<blocks, threads>>>(d_a_mont, d_a_std, m, h_params);
    to_mont_form_kernel<<<blocks, threads>>>(d_b_mont, d_b_std, m, h_params);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Perform NTT and Pointwise Multiplication in Montgomery Domain ---
    ntt_gpu_mont(d_a_mont, m, false, h_params);
    ntt_gpu_mont(d_b_mont, m, false, h_params);

    pointwise_mult_kernel_mont<<<blocks, threads>>>(d_a_mont, d_b_mont, d_a_mont, m, h_params);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Perform Inverse NTT ---
    ntt_gpu_mont(d_a_mont, m, true, h_params);

    // --- Convert Result Back from Montgomery Form ---
    // The result in d_a_mont is now converted back to standard form in d_a_std
    from_mont_form_kernel<<<blocks, threads>>>(d_a_std, d_a_mont, m, h_params);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Copy Final Result to Host ---
    std::vector<unsigned int> ab_padded(m);
    CUDA_CHECK(cudaMemcpy(ab_padded.data(), d_a_std, size_m, cudaMemcpyDeviceToHost));

    for(int i = 0; i < 2 * n - 1; i++) {
        h_ab[i] = ab_padded[i];
    }

    // --- Free GPU Memory ---
    CUDA_CHECK(cudaFree(d_a_std));
    CUDA_CHECK(cudaFree(d_b_std));
    CUDA_CHECK(cudaFree(d_a_mont));
    CUDA_CHECK(cudaFree(d_b_mont));
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
