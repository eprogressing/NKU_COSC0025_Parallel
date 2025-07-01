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
// GPU KERNELS and DEVICE FUNCTIONS (无变动)
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

__global__ void bit_reverse_kernel(int* a, int n) {
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
        int temp = a[i];
        a[i] = a[rev_i];
        a[rev_i] = temp;
    }
}

__global__ void ntt_stage_kernel(int* a, int n, int p, int m, bool is_inverse) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n / 2) return;

    int g = 3;

    long long wm_base = power_gpu(g, (p - 1) / m, p);
    if (is_inverse) {
        wm_base = power_gpu(wm_base, p - 2, p);
    }

    int j = tid % (m / 2);
    int k = (tid / (m / 2)) * m;

    int idx1 = k + j;
    int idx2 = idx1 + m / 2;

    long long w = power_gpu(wm_base, j, p);
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
// GPU WRAPPER FUNCTIONS (已修改)
// =================================================================

// 改动 1: ntt_gpu 函数现在接收一个 threadsPerBlock 参数
void ntt_gpu(int* d_a, int n, int p, bool is_inverse, int threadsPerBlock) {
    // 不再使用硬编码的 threadsPerBlock
    
    int blocks_full = (n + threadsPerBlock - 1) / threadsPerBlock;
    bit_reverse_kernel<<<blocks_full, threadsPerBlock>>>(d_a, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int m = 2; m <= n; m <<= 1) {
        int stage_blocks = (n / 2 + threadsPerBlock - 1) / threadsPerBlock;
        ntt_stage_kernel<<<stage_blocks, threadsPerBlock>>>(d_a, n, p, m, is_inverse);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (is_inverse) {
        normalize_kernel<<<blocks_full, threadsPerBlock>>>(d_a, n, p);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// 改动 2: poly_multiply 函数现在也接收 threadsPerBlock 参数，并传递给所有内核
void poly_multiply(int* h_a, int* h_b, int* h_ab, int n, int p, int threadsPerBlock) {
    int m = 1;
    while (m < 2 * n) { m <<= 1; }
    
    std::vector<int> a_padded(m, 0);
    std::vector<int> b_padded(m, 0);
    for(int i = 0; i < n; i++) {
        a_padded[i] = h_a[i];
        b_padded[i] = h_b[i];
    }

    int *d_a, *d_b;
    size_t size_m = m * sizeof(int);

    CUDA_CHECK(cudaMalloc((void**)&d_a, size_m));
    CUDA_CHECK(cudaMalloc((void**)&d_b, size_m));

    CUDA_CHECK(cudaMemcpy(d_a, a_padded.data(), size_m, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b_padded.data(), size_m, cudaMemcpyHostToDevice));

    // 将 threadsPerBlock 参数传递下去
    ntt_gpu(d_a, m, p, false, threadsPerBlock);
    ntt_gpu(d_b, m, p, false, threadsPerBlock);

    // 改动 3: 点值乘法内核也使用传入的 threadsPerBlock 参数
    int blocks = (m + threadsPerBlock - 1) / threadsPerBlock;
    pointwise_mult_kernel<<<blocks, threadsPerBlock>>>(d_a, d_b, d_a, m, p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 将 threadsPerBlock 参数传递下去
    ntt_gpu(d_a, m, p, true, threadsPerBlock);

    std::vector<int> ab_padded(m);
    CUDA_CHECK(cudaMemcpy(ab_padded.data(), d_a, size_m, cudaMemcpyDeviceToHost));
    
    for(int i = 0; i < 2 * n - 1; i++) {
        h_ab[i] = ab_padded[i];
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
}

// =================================================================
// FILE I/O AND CHECKING (无变动)
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
    for (int i = 0; i < *n; i++){ fin>>a[i]; }
    for (int i = 0; i < *n; i++){ fin>>b[i]; }
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

int a[300000], b[300000], ab[300000];

// =================================================================
// MAIN FUNCTION (已修改)
// =================================================================
int main(int argc, char *argv[])
{
    int test_begin = 1;
    int test_end = 1; 

    // 定义要测试的 threadsPerBlock 值的列表
    std::vector<int> threads_to_test = {64, 128, 256, 512, 1024};

    for(int i = test_begin; i <= test_end; ++i){
        int n_, p_;
        fRead(a, b, &n_, &p_, i);
        if (n_ == 0) continue;

        std::cout << "========================================================" << std::endl;
        std::cout << "Starting Test Case " << i << " (n=" << n_ << ", p=" << p_ << ")" << std::endl;
        std::cout << "========================================================" << std::endl;

        // 循环测试每个 threadsPerBlock 值
        for (int threads : threads_to_test) {
            std::cout << "\n--- Testing with threadsPerBlock = " << threads << " ---" << std::endl;
            
            memset(ab, 0, sizeof(ab));
            auto Start = std::chrono::high_resolution_clock::now();
            
            // 改动 4: 将测试的 threads 值传入 poly_multiply
            poly_multiply(a, b, ab, n_, p_, threads);
            
            CUDA_CHECK(cudaDeviceSynchronize());
            auto End = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double,std::milli> elapsed = End - Start;
            
            fCheck(ab, n_, i);
            std::cout << "Latency: " << elapsed.count() << " (ms)" << std::endl;
            
            // fWrite(ab, n_, i); 
        }
        std::cout << "\n";
    }
    return 0;
}
