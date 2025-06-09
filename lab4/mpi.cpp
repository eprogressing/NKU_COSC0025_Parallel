#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <vector>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <cstdint>
using namespace std;
using int64 = long long;

// 三个模数及其原根
static const int64 mod1 = 7340033;
static const int64 mod2 = 104857601;
static const int64 mod3 = 469762049;
static const int64 G = 3;

// CRT 预处理逆元
static const int64 inv_m1_mod_m2 = []() {
    int64 a = mod1, b = mod2, u = 1, v = 0;
    while (b) {
        int64 t = a / b;
        a -= t * b; swap(a, b);
        u -= t * v; swap(u, v);
    }
    return (u % mod2 + mod2) % mod2;
}();

static const int64 inv_m12_mod_m3 = []() {
    __int128 m12 = (__int128)mod1 * mod2;
    int64 a = (int64)(m12 % mod3);
    int64 b = mod3, u = 1, v = 0;
    while (b) {
        int64 t = a / b;
        a -= t * b; swap(a, b);
        u -= t * v; swap(u, v);
    }
    return (u % mod3 + mod3) % mod3;
}();

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

// Barrett 减模
struct BarrettReducer {
    int64 mod;
    __uint128_t r;
    BarrettReducer(int64 m) : mod(m) {
        r = ((__uint128_t)1 << 64) / mod;
    }
    int64 reduce(__int128 x) const {
        if (x < 0) {
            __int128 k = (-x + mod - 1) / mod;
            x += k * mod;
        }
        __uint128_t v = (unsigned __int128)x;
        __uint128_t q = (v * r) >> 64;
        int64 t = (int64)(v - q * mod);
        if (t >= mod) t -= mod;
        return t;
    }
};

// 单模 NTT（输入输出都是 int64）
void ntt(vector<int64>& a, int64 mod, int type, const BarrettReducer& br) {
    int n = (int)a.size();
    int L = __builtin_ctz(n);
    vector<int> rev(n);
    for (int i = 1; i < n; i++) {
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (L - 1));
    }
    for (int i = 0; i < n; i++) {
        if (i < rev[i]) swap(a[i], a[rev[i]]);
    }
    for (int len = 1; len < n; len <<= 1) {
        int64 wn = modpow(G, (mod - 1) / (len << 1), mod);
        if (type == -1) wn = modpow(wn, mod - 2, mod);
        for (int i = 0; i < n; i += len << 1) {
            int64 w = 1;
            for (int j = 0; j < len; j++) {
                int64 u = a[i + j];
                int64 v = br.reduce((__int128)w * a[i + j + len]);
                int64 x = u + v;
                if (x >= mod) x -= mod;
                int64 y = u - v;
                if (y < 0) y += mod;
                a[i + j] = x;
                a[i + j + len] = y;
                w = br.reduce((__int128)w * wn);
            }
        }
    }
    if (type == -1) {
        int64 invn = modpow(n, mod - 2, mod);
        for (int i = 0; i < n; i++) {
            a[i] = br.reduce((__int128)a[i] * invn);
        }
    }
}

// CRT 合并三路结果到 out，模 P
void crt_merge(const vector<int64>& r1, const vector<int64>& r2, const vector<int64>& r3,
               int *out, int n, int P) {
    int64 m1 = mod1, m2 = mod2, m3 = mod3;
    __int128 m12 = (__int128)m1 * m2;
    int sz = 2 * n - 1;
    for (int i = 0; i < sz; i++) {
        int64 a1 = r1[i], a2 = r2[i], a3 = r3[i];
        int64 t = ((a2 - a1) % m2 + m2) % m2;
        t = (__int128)t * inv_m1_mod_m2 % m2;
        __int128 x12 = a1 + (__int128)t * m1;
        int64 delta = (a3 - (int64)(x12 % m3) + m3) % m3;
        int64 t2 = (__int128)delta * inv_m12_mod_m3 % m3;
        __int128 x = x12 + (__int128)t2 * m12;
        out[i] = int(x % P);
    }
}

// MPI + NTT 多项式乘法接口
void poly_multiply(int *a, int *b, int *out, int n, int P) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int len = 1;
    if (rank == 0) {
        while (len < 2 * n) len <<= 1;
    }
    MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    vector<int64> a_pad(len, 0), b_pad(len, 0);
    if (rank == 0) {
        for (int i = 0; i < n; i++) {
            a_pad[i] = a[i];
            b_pad[i] = b[i];
        }
    }
    MPI_Bcast(a_pad.data(), len, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(b_pad.data(), len, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    if (rank >= 1 && rank <= 3) {
        int64 cur_mod = 0;
        if (rank == 1) cur_mod = mod1;
        else if (rank == 2) cur_mod = mod2;
        else if (rank == 3) cur_mod = mod3;
        BarrettReducer br(cur_mod);
        ntt(a_pad, cur_mod, +1, br);
        ntt(b_pad, cur_mod, +1, br);
        for (int i = 0; i < len; i++) {
            a_pad[i] = br.reduce((__int128)a_pad[i] * b_pad[i]);
        }
        ntt(a_pad, cur_mod, -1, br);
        MPI_Send(a_pad.data(), len, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }
    if (rank == 0) {
        vector<int64> r1(len), r2(len), r3(len);
        MPI_Recv(r1.data(), len, MPI_LONG_LONG, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(r2.data(), len, MPI_LONG_LONG, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(r3.data(), len, MPI_LONG_LONG, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        crt_merge(r1, r2, r3, out, n, P);
    }
}

void fRead(int *a, int *b, int *n, int *p, int id) {
    ifstream fin("/nttdata/" + to_string(id) + ".in");
    fin >> *n >> *p;
    for (int i = 0; i < *n; i++) fin >> a[i];
    for (int i = 0; i < *n; i++) fin >> b[i];
}

void fCheck(int *ab, int n, int id) {
    ifstream fin("/nttdata/" + to_string(id) + ".out");
    for (int i = 0; i < 2 * n - 1; i++) {
        int x; fin >> x;
        if (x != ab[i]) {
            cout << "多项式乘法结果错误" << endl;
            return;
        }
    }
    cout << "多项式乘法结果正确" << endl;
}

void fWrite(int *ab, int n, int id) {
    ofstream fout("files/" + to_string(id) + ".out");
    for (int i = 0; i < 2 * n - 1; i++) {
        fout << ab[i] << '\n';
    }
}

int a[300000], b[300000], ab[300000];

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    for (int id = 0; id <= 3; id++) {
        int n, p;
        fRead(a, b, &n, &p, id);
        memset(ab, 0, sizeof(ab));
        auto t1 = chrono::high_resolution_clock::now();
        poly_multiply(a, b, ab, n, p);
        auto t2 = chrono::high_resolution_clock::now();
        if (MPI::COMM_WORLD.Get_rank() == 0) {
            fCheck(ab, n, id);
            double ms = chrono::duration<double, milli>(t2 - t1).count();
            cout << "average latency for n=" << n << " p=" << p << ": " << ms << " ms" << endl;
            fWrite(ab, n, id);
        }
    }
    MPI_Finalize();
    return 0;
}
