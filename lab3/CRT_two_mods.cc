#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <thread>
#include <vector>
#include <fstream>
#include <iostream>
#include <pthread.h>
using namespace std;
using int64 = long long;

// Two moduli and their primitive root
static const int64 mod1 = 7340033;    // 7,340,033 = 7·2^20+1, primitive root 3
static const int64 mod2 = 104857601;  // 104,857,601 = 25·2^22+1, primitive root 3
static const int64 G = 3;

// Precomputed inverse for CRT
static const int64 inv_m1_mod_m2 = [](){
    int64 a = mod1, b = mod2, u = 1, v = 0;
    while (b) {
        int64 t = a / b;
        a -= t * b; swap(a, b);
        u -= t * v; swap(u, v);
    }
    return (u % mod2 + mod2) % mod2;
}();

// Fast modular exponentiation
int64 modpow(int64 a, int64 e, int64 m) {
    int64 r = 1;
    a %= m;
    while (e) {
        if (e & 1) r = (__int128)r * a % m;
        a = (__int128)a * a % m;
        e >>= 1;
    }
    return r;
}

// Single NTT (type=1 for forward, type=-1 for inverse)
void ntt(int *a, int n, int mod, int type) {
    int L = __builtin_ctz(n);
    vector<int> rev(n);
    rev[0] = 0;
    for (int i = 1; i < n; ++i)
        rev[i] = (rev[i>>1]>>1) | ((i&1)<<(L-1));
    for (int i = 0; i < n; ++i)
        if (i < rev[i]) swap(a[i], a[rev[i]]);

    for (int len = 1; len < n; len <<= 1) {
        int64 wn = modpow(G, (mod - 1) / (len << 1), mod);
        if (type == -1) wn = modpow(wn, mod - 2, mod);
        for (int i = 0; i < n; i += (len << 1)) {
            int64 w = 1;
            for (int j = 0; j < len; ++j) {
                int64 u = a[i+j];
                int64 v = (__int128)w * a[i+j+len] % mod;
                a[i+j]       = (u + v) % mod;
                a[i+j+len]   = (u - v + mod) % mod;
                w = w * wn % mod;
            }
        }
    }
    if (type == -1) {
        int64 inv_n = modpow(n, mod - 2, mod);
        for (int i = 0; i < n; ++i)
            a[i] = (__int128)a[i] * inv_n % mod;
    }
}

struct NTTThreadArg {
    int *a, *b, *result;
    int  n, mod;
};

void* ntt_thread(void* _arg) {
    auto arg = (NTTThreadArg*)_arg;
    int n = arg->n, mod = arg->mod;
    vector<int> A(arg->a, arg->a + n), B(arg->b, arg->b + n);
    ntt(A.data(), n, mod, 1);
    ntt(B.data(), n, mod, 1);
    for (int i = 0; i < n; ++i)
        A[i] = (__int128)A[i] * B[i] % mod;
    ntt(A.data(), n, mod, -1);
    memcpy(arg->result, A.data(), n * sizeof(int));
    return nullptr;
}

// CRT merging for two moduli
void crt_merge(int* r1, int* r2, int* out, int n, int P) {
    int64 m1 = mod1, m2 = mod2;
    for (int i = 0; i < 2 * n - 1; ++i) {
        int64 a1 = r1[i], a2 = r2[i];
        int64 t = ((a2 - a1) % m2 + m2) % m2;
        t = (__int128)t * inv_m1_mod_m2 % m2;
        __int128 x = a1 + (__int128)t * m1;
        out[i] = (int)(x % P);
    }
}

// Polynomial multiplication with two moduli and CRT
void poly_multiply(int *a, int *b, int *out, int n, int P) {
    int len = 1;
    while (len < 2 * n) len <<= 1;
    vector<int> a_pad(len, 0), b_pad(len, 0);
    for (int i = 0; i < n; ++i) {
        a_pad[i] = (a[i] % P + P) % P;
        b_pad[i] = (b[i] % P + P) % P;
    }
    vector<int> r1(len), r2(len);
    NTTThreadArg args[2] = {
        {a_pad.data(), b_pad.data(), r1.data(), len, (int)mod1},
        {a_pad.data(), b_pad.data(), r2.data(), len, (int)mod2}
    };
    pthread_t threads[2];
    for (int i = 0; i < 2; ++i)
        pthread_create(&threads[i], nullptr, ntt_thread, &args[i]);
    for (int i = 0; i < 2; ++i)
        pthread_join(threads[i], nullptr);
    crt_merge(r1.data(), r2.data(), out, n, P);
}

void fRead(int *a, int *b, int *n, int *p, int input_id){
    string strin = string("/nttdata/") + to_string(input_id) + ".in";
    ifstream fin(strin);
    fin >> *n >> *p;
    for (int i = 0; i < *n; ++i) fin >> a[i];
    for (int i = 0; i < *n; ++i) fin >> b[i];
}

void fCheck(int *ab, int n, int input_id){
    string strout = string("/nttdata/") + to_string(input_id) + ".out";
    ifstream fin(strout);
    for (int i = 0; i < n * 2 - 1; ++i) {
        int x; fin >> x;
        if (x != ab[i]) { cout << "多项式乘法结果错误" << endl; return; }
    }
    cout << "多项式乘法结果正确" << endl;
}

void fWrite(int *ab, int n, int input_id){
    string strout = string("files/") + to_string(input_id) + ".out";
    ofstream fout(strout);
    for (int i = 0; i < n * 2 - 1; ++i) fout << ab[i] << '\n';
}

int a[300000], b[300000], ab[300000];
int main(int argc, char *argv[]){
    int test_begin = 0, test_end = 3;
    for(int i = test_begin; i <= test_end; ++i){
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
