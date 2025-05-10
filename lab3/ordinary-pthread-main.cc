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

// 快速幂
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

// 找 P 的一个原根
int get_primitive_root(int P) {
    int phi = P - 1, n = phi;
    vector<int> fac;
    for (int i = 2; i * i <= n; ++i) {
        if (n % i == 0) {
            fac.push_back(i);
            while (n % i == 0) n /= i;
        }
    }
    if (n > 1) fac.push_back(n);
    for (int g = 2; g < P; ++g) {
        bool ok = true;
        for (int f : fac) {
            if (modpow(g, phi / f, P) == 1) {
                ok = false; break;
            }
        }
        if (ok) return g;
    }
    return -1;
}

// 单次 NTT（type=1 正变换，type=-1 反变换）
void ntt(int *a, int n, int P, int G, int type) {
    int L = __builtin_ctz(n);
    vector<int> rev(n);
    rev[0] = 0;
    for (int i = 1; i < n; ++i)
        rev[i] = (rev[i>>1]>>1) | ((i&1)<<(L-1));
    for (int i = 0; i < n; ++i)
        if (i < rev[i]) swap(a[i], a[rev[i]]);

    for (int len = 1; len < n; len <<= 1) {
        int64 wn = modpow(G, (P - 1) / (len << 1), P);
        if (type == -1) wn = modpow(wn, P - 2, P);
        for (int i = 0; i < n; i += (len << 1)) {
            int64 w = 1;
            for (int j = 0; j < len; ++j) {
                int64 u = a[i+j];
                int64 v = (__int128)w * a[i+j+len] % P;
                a[i+j]       = (u + v) % P;
                a[i+j+len]   = (u - v + P) % P;
                w = w * wn % P;
            }
        }
    }
    if (type == -1) {
        int64 inv_n = modpow(n, P - 2, P);
        for (int i = 0; i < n; ++i)
            a[i] = (__int128)a[i] * inv_n % P;
    }
}

struct NTTThreadArg {
    int *a;
    int  n;
    int  P, G;
    int  type;  // 1 or -1
};

void* ntt_thread(void* _arg) {
    auto arg = (NTTThreadArg*)_arg;
    ntt(arg->a, arg->n, arg->P, arg->G, arg->type);
    return nullptr;
}

// 基于 pthread 的多线程 NTT 多项式乘法（无需 CRT）
void poly_multiply(int *a, int *b, int *out, int n, int P) {
    int len = 1;
    while (len < 2 * n) len <<= 1;
    vector<int> A(len, 0), B(len, 0);
    for (int i = 0; i < n; ++i) {
        A[i] = (a[i] % P + P) % P;
        B[i] = (b[i] % P + P) % P;
    }

    // 找到原根
    int G = get_primitive_root(P);

    // 两线程正向 NTT
    pthread_t t1, t2;
    NTTThreadArg arg1{A.data(), len, P, G, 1};
    NTTThreadArg arg2{B.data(), len, P, G, 1};
    pthread_create(&t1, nullptr, ntt_thread, &arg1);
    pthread_create(&t2, nullptr, ntt_thread, &arg2);
    pthread_join(t1, nullptr);
    pthread_join(t2, nullptr);

    // 点乘
    for (int i = 0; i < len; ++i)
        A[i] = (int)((int64)A[i] * B[i] % P);

    // 一线程反向 NTT，注意这里使用局部变量 arg3
    NTTThreadArg arg3{A.data(), len, P, G, -1};
    pthread_create(&t1, nullptr, ntt_thread, &arg3);
    pthread_join(t1, nullptr);

    // 拷贝结果
    for (int i = 0; i < 2 * n - 1; ++i)
        out[i] = A[i];
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
