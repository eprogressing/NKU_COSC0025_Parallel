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
// HELPER & PRECOMPUTATION
// =================================================================

// Host-side power function for precomputation
long long power_host(long long base, long long exp, long long mod) {
    long long res = 1;
    base %= mod;
    while (exp > 0) {
        if (exp % 2 == 1) res = (res * base) % mod;
        base = (base * base) % mod;
        exp /= 2;
    }
    return res;
}

// Precomputes all necessary twiddle factors on the host
void precompute_twiddle_factors(std::vector<int>& wn, int n, int p, bool is_inverse) {
    int g = 3; // Primitive root
    if (n <= 1) return;
    // Total twiddles needed for all stages is sum of m/2 for m=2,4..n, which is n-1
    wn.resize(n - 1);
    
    int offset = 0;
    for (int m = 2; m <= n; m <<= 1) {
        long long wm_base = power_host(g, (p - 1) / m, p);
        if (is_inverse) {
            wm_base = power_host(wm_base, p - 2, p);
        }

        long long w = 1;
        for (int j = 0; j < m / 2; j++) {
            wn[offset + j] = w;
            w = (w * wm_base) % p;
        }
        offset += m / 2;
    }
}


// =================================================================
// GPU KERNELS and DEVICE FUNCTIONS
// =================================================================

__device__ long long power_gpu(long long base, long long exp, long long mod) {
    long long res = 1;
    base %= mod;
    while (exp > 0) {
        if (exp % 2 == 1) res = (res * base) % mod;
        base = (base * base) % mod;
        exp /= 2;
    }
    return res;
}

// 修复后的位反转内核
__global__ void bit_reverse_kernel(int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // 计算整数log2(n)
    int logn = 0;
    int temp_n = n;
    while (temp_n > 1) {
        logn++;
        temp_n >>= 1;
    }
    
    // 位反转计算
    int rev_i = 0;
    int temp = i;
    for (int j = 0; j < logn; j++) {
        rev_i = (rev_i << 1) | (temp & 1);
        temp >>= 1;
    }

    // 确保反转索引在范围内
    if (rev_i < n && i < rev_i) {
        int temp_val = a[i];
        a[i] = a[rev_i];
        a[rev_i] = temp_val;
    }
}

// KERNEL 1: Fast version for small stages using Shared Memory
__global__ void ntt_stage_kernel_shared(int* a, const int* wn_global, int m, int stage_offset, int p) {
    extern __shared__ int s_mem[];
    int* s_a = s_mem;
    int* s_w = &s_mem[m];

    int tid = threadIdx.x;          // Thread index in block: 0 to m/2 - 1
    int k = blockIdx.x * m;         // Base index of the butterfly group in global memory

    s_a[tid]       = a[k + tid];
    s_a[tid + m/2] = a[k + tid + m/2];
    s_w[tid]       = wn_global[stage_offset + tid];

    __syncthreads(); 

    long long w = s_w[tid];
    long long u = s_a[tid];
    long long v = s_a[tid + m/2];
    long long t = (w * v) % p;

    s_a[tid]       = (u + t) % p;
    s_a[tid + m/2] = (u - t + p) % p;

    __syncthreads(); 

    a[k + tid]       = s_a[tid];
    a[k + tid + m/2] = s_a[tid + m/2];
}

// KERNEL 2: Slower, safe version for large stages using Global Memory
__global__ void ntt_stage_kernel_global(int* a, const int* wn_global, int n, int m, int stage_offset, int p) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n / 2) return; 

    int j = tid % (m / 2); 
    int k = (tid / (m / 2)) * m; 

    int idx1 = k + j;
    int idx2 = idx1 + m / 2;

    long long w = wn_global[stage_offset + j];
    long long t = (w * a[idx2]) % p;
    long long u = a[idx1];
    
    a[idx1] = (u + t) % p;
    a[idx2] = (u - t + p) % p;
}


__global__ void pointwise_mult_kernel(int* a, int* b, int* ab, int n, int p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        ab[i] = (1LL * a[i] * b[i]) % p;
    }
}

__global__ void normalize_kernel(int* a, int n, int p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        long long n_inv = power_gpu(n, p - 2, p);
        a[i] = (1LL * a[i] * n_inv) % p;
    }
}

// =================================================================
// GPU WRAPPER FUNCTION (The new poly_multiply)
// =================================================================
void ntt_gpu(int* d_a, int* d_wn, int n, int p, bool is_inverse) {
    int threadsPerBlock = 256;
    
    int blocks_full = (n + threadsPerBlock - 1) / threadsPerBlock;
    bit_reverse_kernel<<<blocks_full, threadsPerBlock>>>(d_a, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int stage_offset = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    for (int m = 2; m <= n; m <<= 1) {
        int threads_needed_for_shared = m / 2;
        size_t shared_mem_size = (size_t)(m + m/2) * sizeof(int);

        // HYBRID STRATEGY: Check if the stage is suitable for the shared memory kernel
        if (threads_needed_for_shared <= prop.maxThreadsPerBlock && shared_mem_size <= prop.sharedMemPerBlock) {
            // Use fast shared memory kernel
            int blocks = n / m;
            ntt_stage_kernel_shared<<<blocks, threads_needed_for_shared, shared_mem_size>>>(d_a, d_wn, m, stage_offset, p);
        } else {
            // Use slower but safe global memory kernel
            int blocks = (n / 2 + threadsPerBlock - 1) / threadsPerBlock;
            ntt_stage_kernel_global<<<blocks, threadsPerBlock>>>(d_a, d_wn, n, m, stage_offset, p);
        }
        
        stage_offset += m / 2;
    }

    if (is_inverse) {
        normalize_kernel<<<blocks_full, threadsPerBlock>>>(d_a, n, p);
    }
}

void poly_multiply(int* h_a, int* h_b, int* h_ab, int n, int p) {
    int m = 1;
    while (m < 2 * n) { m <<= 1; }
    
    std::vector<int> a_padded(m, 0);
    std::vector<int> b_padded(m, 0);
    for(int i = 0; i < n; i++) {
        a_padded[i] = h_a[i];
        b_padded[i] = h_b[i];
    }

    std::vector<int> h_wn_fwd, h_wn_inv;
    precompute_twiddle_factors(h_wn_fwd, m, p, false);
    precompute_twiddle_factors(h_wn_inv, m, p, true);

    int *d_a, *d_b, *d_wn_fwd, *d_wn_inv;
    size_t size_m = m * sizeof(int);
    size_t size_wn = (m > 1) ? (m - 1) * sizeof(int) : 0;

    CUDA_CHECK(cudaMalloc((void**)&d_a, size_m));
    CUDA_CHECK(cudaMalloc((void**)&d_b, size_m));
    if (size_wn > 0) {
        CUDA_CHECK(cudaMalloc((void**)&d_wn_fwd, size_wn));
        CUDA_CHECK(cudaMalloc((void**)&d_wn_inv, size_wn));
    }

    CUDA_CHECK(cudaMemcpy(d_a, a_padded.data(), size_m, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b_padded.data(), size_m, cudaMemcpyHostToDevice));
    if (size_wn > 0) {
        CUDA_CHECK(cudaMemcpy(d_wn_fwd, h_wn_fwd.data(), size_wn, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_wn_inv, h_wn_inv.data(), size_wn, cudaMemcpyHostToDevice));
    }

    // --- GPU Execution ---
    ntt_gpu(d_a, d_wn_fwd, m, p, false);
    ntt_gpu(d_b, d_wn_fwd, m, p, false);

    int threads = 256;
    int blocks = (m + threads - 1) / threads;
    pointwise_mult_kernel<<<blocks, threads>>>(d_a, d_b, d_a, m, p);
    
    ntt_gpu(d_a, d_wn_inv, m, p, true);
    
    // Single sync after all kernels are launched
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Copy result back and clean up ---
    std::vector<int> ab_padded(m);
    CUDA_CHECK(cudaMemcpy(ab_padded.data(), d_a, size_m, cudaMemcpyDeviceToHost));
    
    for(int i = 0; i < 2 * n - 1; i++) {
        h_ab[i] = ab_padded[i];
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    if (size_wn > 0) {
        CUDA_CHECK(cudaFree(d_wn_fwd));
        CUDA_CHECK(cudaFree(d_wn_inv));
    }
}

// =================================================================
// USER'S ORIGINAL FRAMEWORK (UNCHANGED, except for file paths)
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

void fCheck(int *ab, int n, int p, int input_id){
    std::string str1 = "nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";

    std::ifstream fin(strout);
    if (!fin.is_open()) {
        std::cerr << "Error opening output check file: " << strout << std::endl;
        return;
    }
    bool correct = true;
    for (int i = 0; i < n * 2 - 1; i++){
        int x;
        fin>>x;
        if (!fin) {
             std::cout << "Error reading from check file or file ended prematurely." << std::endl;
             correct = false;
             break;
        }
        if(x != ab[i]){
            std::cout<<"多项式乘法结果错误 at index " << i << ". Expected " << x << ", got " << ab[i] << std::endl;
            correct = false;
        }
    }
    if (correct) {
        std::cout<<"多项式乘法结果正确"<<std::endl;
    }
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

int a[300000], b[300000], ab[300000];

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
        
        auto End = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> elapsed = End - Start;
        ans += elapsed.count();
        
        fCheck(ab, n_, p_, i);
        std::cout<<"Latency for n = "<<n_<<", p = "<<p_<<" : "<<ans<<" (ms) "<<std::endl;
        
        fWrite(ab, n_, i);
    }
    return 0;
}
