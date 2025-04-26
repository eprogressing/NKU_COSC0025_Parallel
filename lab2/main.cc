#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <omp.h>
#include <arm_neon.h>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <algorithm>

using namespace std;

typedef uint32_t u32;
typedef uint64_t u64;

static void* aligned_malloc(size_t alignment, size_t size) {
    void* ptr = nullptr;
#ifdef _MSC_VER
    ptr = _aligned_malloc(size, alignment);
#else
    if (posix_memalign(&ptr, alignment, size) != 0)
        ptr = nullptr;
#endif
    return ptr;
}

static void aligned_free(void* ptr) {
#ifdef _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static u64 mod_inverse(u64 a, u64 mod) {
    u64 old_r = a, r = mod;
    int64_t old_s = 1, s = 0;

    while (r != 0) {
        u64 q = old_r / r;
        u64 tmp_r = r;
        r = old_r - q * r;
        old_r = tmp_r;

        int64_t tmp_s = s;
        s = old_s - (int64_t)q * s;
        old_s = tmp_s;
    }
    return (old_s < 0) ? (old_s + mod) : old_s;
}

static u32 pow_mod(u32 base, u32 exp, u32 mod) {
    u32 result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1)
            result = (u64)result * base % mod;
        base = (u64)base * base % mod;
        exp >>= 1;
    }
    return result;
}

static u32* precompute_wn(u32 mod, int limit, u32 root) {
    int logn = 0;
    while ((1 << logn) < limit) logn++;
    u32* wn = (u32*)aligned_malloc(32, sizeof(u32) * limit);
    assert(wn);

    int offset = 0;
    for (int stage = 0; stage < logn; stage++) {
        int mid = 1 << stage;
        u32 w = pow_mod(root, (mod - 1) >> (stage + 1), mod);
        wn[offset] = 1;
        for (int k = 1; k < mid; k++) {
            wn[offset + k] = (u64)wn[offset + k - 1] * w % mod;
        }
        offset += mid;
    }
    return wn;
}

static u32* precompute_wn_montgomery(u32 mod, u32 R_mod, const u32* wn, int limit) {
    u32* wn_mont = (u32*)aligned_malloc(32, sizeof(u32) * limit);
    assert(wn_mont);
    for (int i = 0; i < limit; i++) {
        wn_mont[i] = (u64)wn[i] * R_mod % mod;
    }
    return wn_mont;
}

static inline u32 montgomery_mul_scalar(u32 a, u32 b, u32 mod, u32 mont_factor) {
    u64 t = (u64)a * b;
    u32 m = (u32)t * mont_factor;
    u64 u = (t + (u64)m * mod) >> 32;
    return (u32)u >= mod ? (u32)u - mod : (u32)u;
}

static inline uint32x4_t montgomery_mul_neon(uint32x4_t a, uint32x4_t b, u32 mod, u32 mont_factor) {
    uint32x2_t a_lo = vget_low_u32(a);
    uint32x2_t a_hi = vget_high_u32(a);
    uint32x2_t b_lo = vget_low_u32(b);
    uint32x2_t b_hi = vget_high_u32(b);

    uint64x2_t t_lo = vmull_u32(a_lo, b_lo);
    uint64x2_t t_hi = vmull_u32(a_hi, b_hi);

    uint32x4_t t_low = vcombine_u32(vmovn_u64(t_lo), vmovn_u64(t_hi));
    uint32x4_t m_vec = vmulq_n_u32(t_low, mont_factor);

    uint32x2_t mod_vec2 = vdup_n_u32(mod);
    uint64x2_t product_lo = vmull_u32(vget_low_u32(m_vec), mod_vec2);
    uint64x2_t product_hi = vmull_u32(vget_high_u32(m_vec), mod_vec2);

    uint64x2_t sum_lo = vaddq_u64(t_lo, product_lo);
    uint64x2_t sum_hi = vaddq_u64(t_hi, product_hi);

    uint32x4_t u_vec = vcombine_u32(
        vshrn_n_u64(sum_lo, 32),
        vshrn_n_u64(sum_hi, 32)
    );

    uint32x4_t mod_vec4 = vdupq_n_u32(mod);
    uint32x4_t mask = vcgeq_u32(u_vec, mod_vec4);
    uint32x4_t sub = vsubq_u32(u_vec, mod_vec4);
    return vbslq_u32(mask, sub, u_vec);
}

static void ntt_butterfly_neon(u32* a, int limit, const u32* wn_mont, u32 mod, u32 mont_factor) {
    int logn = 0;
    while ((1 << logn) < limit) logn++;
    int offset = 0;

    for (int stage = 0; stage < logn; stage++) {
        int mid = 1 << stage;
        for (int j = 0; j < limit; j += (mid << 1)) {
            if (mid >= 4) {
                for (int k = 0; k < mid; k += 4) {
                    uint32x4_t x = vld1q_u32(a + j + k);
                    uint32x4_t y = vld1q_u32(a + j + mid + k);
                    uint32x4_t w = vld1q_u32(wn_mont + offset + k);

                    uint32x4_t yw = montgomery_mul_neon(y, w, mod, mont_factor);

                    uint32x4_t sum = vaddq_u32(x, yw);
                    sum = vsubq_u32(sum, vdupq_n_u32(mod));
                    uint32x4_t sum_mask = vcgeq_u32(x, vsubq_u32(vdupq_n_u32(mod), yw));
                    sum = vbslq_u32(sum_mask, sum, vaddq_u32(x, yw));

                    uint32x4_t diff = vsubq_u32(x, yw);
                    diff = vaddq_u32(diff, vdupq_n_u32(mod));
                    uint32x4_t diff_mask = vcgeq_u32(diff, vdupq_n_u32(mod));
                    diff = vbslq_u32(diff_mask, vsubq_u32(diff, vdupq_n_u32(mod)), diff);

                    vst1q_u32(a + j + k, sum);
                    vst1q_u32(a + j + mid + k, diff);
                }
            } else {
                for (int k = 0; k < mid; k++) {
                    u32 x = a[j + k];
                    u32 y = a[j + mid + k];
                    y = montgomery_mul_scalar(y, wn_mont[offset + k], mod, mont_factor);
                    u32 u = (x + y) % mod;
                    u32 v = (x + mod - y) % mod;
                    a[j + k] = u;
                    a[j + mid + k] = v;
                }
            }
        }
        offset += mid;
    }
}

static void bit_reverse(u32* a, int n) {
    int logn = 0;
    while ((1 << logn) < n) logn++;
    for (int i = 0; i < n; ++i) {
        int rev = 0;
        for (int j = 0; j < logn; j++) {
            rev = (rev << 1) | ((i >> j) & 1);
        }
        if (i < rev) {
            std::swap(a[i], a[rev]);
        }
    }
}

void poly_multiply(int *a, int *b, int *ab, int n, int p) {
    int m = 1;
    while (m < 2 * n - 1) m <<= 1;

    const u64 R = 1ULL << 32;
    const u32 R_mod = R % p;
    const u64 inv = mod_inverse(p, R);
    const u32 mont_factor = (inv == 0) ? 0 : (u32)(R - inv);

    u32 *ta = (u32*)aligned_malloc(32, m * sizeof(u32));
    u32 *tb = (u32*)aligned_malloc(32, m * sizeof(u32));
    memset(ta, 0, m * sizeof(u32));
    memset(tb, 0, m * sizeof(u32));

    for (int i = 0; i < n; ++i) {
        ta[i] = (u64)a[i] * R_mod % p;
        tb[i] = (u64)b[i] * R_mod % p;
    }

    // 正向 NTT 前进行 bit-reverse
    bit_reverse(ta, m);
    bit_reverse(tb, m);

    // 正向 NTT（原根 3）
    u32 *wn = precompute_wn(p, m, 3);
    u32 *wn_mont = precompute_wn_montgomery(p, R_mod, wn, m);
    ntt_butterfly_neon(ta, m, wn_mont, p, mont_factor);
    ntt_butterfly_neon(tb, m, wn_mont, p, mont_factor);

    // 点乘
    for (int i = 0; i < m; ++i) {
        ta[i] = montgomery_mul_scalar(ta[i], tb[i], p, mont_factor);
    }

    // 逆 NTT（原根逆）
    u32 inv_g = mod_inverse(3, p);
    u32* iwn = precompute_wn(p, m, inv_g);
    u32* iwn_mont = precompute_wn_montgomery(p, R_mod, iwn, m);

    // 逆 NTT 前也需 bit-reverse
    bit_reverse(ta, m);
    ntt_butterfly_neon(ta, m, iwn_mont, p, mont_factor);

    // 归一化 & 转换出 Montgomery
    const u32 inv_m = mod_inverse(m, p);
    const u32 inv_m_mont = (u64)inv_m * R_mod % p;
    for (int i = 0; i < m; ++i) {
        // 先乘 inv_m，再乘 1 去掉 Montgomery
        ta[i] = montgomery_mul_scalar(ta[i], inv_m_mont, p, mont_factor);
        ta[i] = montgomery_mul_scalar(ta[i], 1, p, mont_factor);
    }

    // 输出 2n-1 项
    for (int i = 0; i < 2 * n - 1; ++i) {
        ab[i] = ta[i] % p;
    }

    aligned_free(ta);
    aligned_free(tb);
    aligned_free(wn);
    aligned_free(wn_mont);
    aligned_free(iwn);
    aligned_free(iwn_mont);
}

void fRead(int *a, int *b, int *n, int *p, int input_id){
    // 数据输入函数
    std::string str1 = "/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strin = str1 + str2 + ".in";
    char data_path[strin.size() + 1];
    std::copy(strin.begin(), strin.end(), data_path);
    data_path[strin.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    fin>>*n>>*p;
    for (int i = 0; i < *n; i++){
        fin>>a[i];
    }
    for (int i = 0; i < *n; i++){   
        fin>>b[i];
    }
}

void fCheck(int *ab, int n, int input_id){
    // 判断多项式乘法结果是否正确
    std::string str1 = "/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char data_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), data_path);
    data_path[strout.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    for (int i = 0; i < n * 2 - 1; i++){
        int x;
        fin>>x;
        if(x != ab[i]){
            std::cout<<"多项式乘法结果错误"<<std::endl;
            return;
        }
    }
    std::cout<<"多项式乘法结果正确"<<std::endl;
    return;
}

void fWrite(int *ab, int n, int input_id){
    // 数据输出函数, 可以用来输出最终结果, 也可用于调试时输出中间数组
    std::string str1 = "files/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char output_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), output_path);
    output_path[strout.size()] = '\0';
    std::ofstream fout;
    fout.open(output_path, std::ios::out);
    for (int i = 0; i < n * 2 - 1; i++){
        fout<<ab[i]<<'\n';
    }
}

int a[300000], b[300000], ab[300000];

int main(int argc, char *argv[])
{

    // 保证输入的所有模数的原根均为 3, 且模数都能表示为 a \times 4 ^ k + 1 的形式
    // 输入模数分别为 7340033 104857601 469762049 263882790666241
    // 第四个模数超过了整型表示范围, 如果实现此模数意义下的多项式乘法需要修改框架
    // 对第四个模数的输入数据不做必要要求, 如果要自行探索大模数 NTT, 请在完成前三个模数的基础代码及优化后实现大模数 NTT
    // 输入文件共五个, 第一个输入文件 n = 4, 其余四个文件分别对应四个模数, n = 131072
    // 在实现快速数论变化前, 后四个测试样例运行时间较久, 推荐调试正确性时只使用输入文件 1
    int test_begin = 0;
    int test_end = 3;
    for(int i = test_begin; i <= test_end; ++i){
        long double ans = 0;
        int n_, p_;
        fRead(a, b, &n_, &p_, i);
        memset(ab,0,sizeof(ab));
        auto Start = std::chrono::high_resolution_clock::now();
        // TODO : 将 poly_multiply 函数替换成你写的 ntt
        poly_multiply(a, b, ab, n_, p_);
        auto End = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::ratio<1,1000>>elapsed = End - Start;
        ans += elapsed.count();
        fCheck(ab, n_, i);
        std::cout<<"average latency for n = "<<n_<<" p = "<<p_<<" : "<<ans<<" (us) "<<std::endl;
        // 可以使用 fWrite 函数将 ab 的输出结果打印到 files 文件夹下
        // 禁止使用 cout 一次性输出大量文件内容
        fWrite(ab, n_, i);
    }
    return 0;
}
