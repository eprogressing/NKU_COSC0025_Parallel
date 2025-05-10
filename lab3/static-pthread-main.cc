#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <vector>
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <atomic>

using namespace std;
using int64 = long long;

// 三个模数及其原根
static const int64 mod1 = 7340033;
static const int64 mod2 = 104857601;
static const int64 mod3 = 469762049;
static const int64 G = 3;

// 预计算 CRT 所需逆元
static const int64 inv_m1_mod_m2 = []() {
    int64 a = mod1;
    int64 b = mod2;
    int64 u = 1;
    int64 v = 0;
    while (b) {
        int64 t = a / b;
        a -= t * b;
        swap(a, b);
        u -= t * v;
        swap(u, v);
    }
    return (u % mod2 + mod2) % mod2;
}();

static const int64 inv_m12_mod_m3 = []() {
    int64 m12 = (mod1 * mod2) % mod3;
    int64 a = m12;
    int64 b = mod3;
    int64 u = 1;
    int64 v = 0;
    while (b) {
        int64 t = a / b;
        a -= t * b;
        swap(a, b);
        u -= t * v;
        swap(u, v);
    }
    return (u % mod3 + mod3) % mod3;
}();

// 快速幂模运算
int64 modpow(int64 a, int64 e, int64 m) {
    int64 r = 1;
    a %= m;
    while (e) {
        if (e & 1) {
            r = (__int128)r * a % m;
        }
        a = (__int128)a * a % m;
        e >>= 1;
    }
    return r;
}

// 单模 NTT 变换
void ntt(int *a, int n, int mod, int type) {
    int L = __builtin_ctz(n);
    vector<int> rev(n);
    rev[0] = 0;
    for (int i = 1; i < n; ++i) {
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (L - 1));
    }

    for (int i = 0; i < n; ++i) {
        if (i < rev[i]) {
            swap(a[i], a[rev[i]]);
        }
    }

    for (int len = 1; len < n; len <<= 1) {
        int64 wn = modpow(G, (mod - 1) / (len << 1), mod);
        if (type == -1) {
            wn = modpow(wn, mod - 2, mod);
        }
        for (int i = 0; i < n; i += len << 1) {
            int64 w = 1;
            for (int j = 0; j < len; ++j) {
                int64 u = a[i + j];
                int64 v = (__int128)w * a[i + j + len] % mod;
                a[i + j] = (u + v) % mod;
                a[i + j + len] = (u - v + mod) % mod;
                w = (__int128)w * wn % mod;
            }
        }
    }

    if (type == -1) {
        int64 inv_n = modpow(n, mod - 2, mod);
        for (int i = 0; i < n; ++i) {
            a[i] = (__int128)a[i] * inv_n % mod;
        }
    }
}

// 全局并行缓冲区
static vector<int> g_a_pad;
static vector<int> g_b_pad;
static int        g_len;
static vector<int> g_res1;
static vector<int> g_res2;
static vector<int> g_res3;
static atomic<bool> ready1(false);
static atomic<bool> ready2(false);
static atomic<bool> ready3(false);
static atomic<bool> done1(false);
static atomic<bool> done2(false);
static atomic<bool> done3(false);

// 驻留线程：mod1
void* worker1(void*) {
    while (true) {
        if (ready1) {
            vector<int> A = g_a_pad;
            vector<int> B = g_b_pad;
            ntt(A.data(), g_len, mod1, 1);
            ntt(B.data(), g_len, mod1, 1);
            for (int i = 0; i < g_len; ++i) {
                A[i] = (__int128)A[i] * B[i] % mod1;
            }
            ntt(A.data(), g_len, mod1, -1);
            g_res1 = A;
            done1 = true;
            ready1 = false;
        }
    }
    return nullptr;
}

// 驻留线程：mod2
void* worker2(void*) {
    while (true) {
        if (ready2) {
            vector<int> A = g_a_pad;
            vector<int> B = g_b_pad;
            ntt(A.data(), g_len, mod2, 1);
            ntt(B.data(), g_len, mod2, 1);
            for (int i = 0; i < g_len; ++i) {
                A[i] = (__int128)A[i] * B[i] % mod2;
            }
            ntt(A.data(), g_len, mod2, -1);
            g_res2 = A;
            done2 = true;
            ready2 = false;
        }
    }
    return nullptr;
}

// 驻留线程：mod3
void* worker3(void*) {
    while (true) {
        if (ready3) {
            vector<int> A = g_a_pad;
            vector<int> B = g_b_pad;
            ntt(A.data(), g_len, mod3, 1);
            ntt(B.data(), g_len, mod3, 1);
            for (int i = 0; i < g_len; ++i) {
                A[i] = (__int128)A[i] * B[i] % mod3;
            }
            ntt(A.data(), g_len, mod3, -1);
            g_res3 = A;
            done3 = true;
            ready3 = false;
        }
    }
    return nullptr;
}

// 双步 CRT 合并
void crt_merge(int* r1, int* r2, int* r3, int* out, int n, int P) {
    int64 m1 = mod1;
    int64 m2 = mod2;
    int64 m3 = mod3;
    int64 m12 = m1 * m2;

    for (int i = 0; i < 2 * n - 1; ++i) {
        int64 a1 = r1[i];
        int64 a2 = r2[i];
        int64 a3 = r3[i];

        int64 t = ((a2 - a1) % m2 + m2) % m2;
        t = (__int128)t * inv_m1_mod_m2 % m2;
        __int128 x12 = a1 + (__int128)t * m1;

        int64 t2 = ((a3 - (int64)(x12 % m3)) % m3 + m3) % m3;
        t2 = (__int128)t2 * inv_m12_mod_m3 % m3;
        __int128 x = x12 + (__int128)t2 * m12;

        out[i] = (int)(x % P);
    }
}

// 多项式乘法接口
void poly_multiply(int *a, int *b, int *out, int n, int P) {
    g_len = 1;
    while (g_len < 2 * n) {
        g_len <<= 1;
    }

    g_a_pad.assign(g_len, 0);
    g_b_pad.assign(g_len, 0);
    for (int i = 0; i < n; ++i) {
        g_a_pad[i] = a[i];
        g_b_pad[i] = b[i];
    }

    g_res1.resize(g_len);
    g_res2.resize(g_len);
    g_res3.resize(g_len);

    done1 = false;
    done2 = false;
    done3 = false;
    ready1 = true;
    ready2 = true;
    ready3 = true;

    while (!done1 || !done2 || !done3);

    crt_merge(g_res1.data(), g_res2.data(), g_res3.data(), out, n, P);
}

// I/O 辅助
void fRead(int *a, int *b, int *n, int *p, int id) {
    ifstream fin(string("/nttdata/") + to_string(id) + ".in");
    fin >> *n >> *p;
    for (int i = 0; i < *n; ++i) {
        fin >> a[i];
    }
    for (int i = 0; i < *n; ++i) {
        fin >> b[i];
    }
}

void fCheck(int *ab, int n, int id) {
    ifstream fin(string("/nttdata/") + to_string(id) + ".out");
    for (int i = 0; i < 2 * n - 1; ++i) {
        int x;
        fin >> x;
        if (x != ab[i]) {
            cout << "多项式乘法结果错误\n";
            return;
        }
    }
    cout << "多项式乘法结果正确\n";
}

void fWrite(int *ab, int n, int id) {
    ofstream fout(string("files/") + to_string(id) + ".out");
    for (int i = 0; i < 2 * n - 1; ++i) {
        fout << ab[i] << '\n';
    }
}

int a[300000], b[300000], ab[300000];

int main(int argc, char *argv[]) {
    pthread_t t1, t2, t3;
    pthread_create(&t1, nullptr, worker1, nullptr);
    pthread_create(&t2, nullptr, worker2, nullptr);
    pthread_create(&t3, nullptr, worker3, nullptr);

    int test_begin = 0, test_end = 3;
    for (int i = test_begin; i <= test_end; ++i) {
        long double ans = 0;
        int n_, p_;
        fRead(a, b, &n_, &p_, i);
        memset(ab, 0, sizeof(ab));

        auto Start = chrono::high_resolution_clock::now();
        poly_multiply(a, b, ab, n_, p_);
        auto End = chrono::high_resolution_clock::now();
        ans += chrono::duration<double, milli>(End - Start).count();

        fCheck(ab, n_, i);
        cout << "average latency for n = " << n_ << " p = " << p_ << " : " << ans << " (us)" << endl;
        fWrite(ab, n_, i);
    }

    return 0;
}
