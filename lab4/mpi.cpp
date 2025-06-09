#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <vector>
#include <fstream>
#include <iostream>
#include <mpi.h> // Include MPI header

using namespace std;
using int64 = long long;

// Three moduli and their primitive roots
static const int64 mod1 = 7340033;
static const int64 mod2 = 104857601;
static const int64 mod3 = 469762049;
static const int64 G = 3;

// Pre-computed inverse elements for CRT
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
    int64 m12 = (mod1 * mod2) % mod3;
    int64 a = m12, b = mod3, u = 1, v = 0;
    while (b) {
        int64 t = a / b;
        a -= t * b; swap(a, b);
        u -= t * v; swap(u, v);
    }
    return (u % mod3 + mod3) % mod3;
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

// Barrett Modular Reduction
// Implements the optimization described in the prompt.
// For a modulus q, it calculates x mod q as x - floor(x*r / 2^k) * q
// Here, we use k=64.
struct BarrettReducer {
    int64 mod;
    __uint128_t r;

    BarrettReducer(int64 m) : mod(m) {
        // Pre-compute r = floor(2^128 / mod) for 64-bit inputs
        // This is a more stable way to calculate r for k=64.
        r = ((__uint128_t)1 << 64) / mod;
    }

    // Reduces a 64-bit integer
    int64 reduce(int64 x) const {
        return x % mod; // Standard reduction for smaller numbers
    }

    // Reduces a 128-bit integer (result of multiplication)
    int64 reduce(__int128 x) const {
        // The core of Barrett Reduction:
        // 1. Multiply by pre-computed r
        // 2. Shift to get the quotient estimate
        // 3. Subtract to get the remainder
        __uint128_t val = x;
        __uint128_t q_estimate = (val * r) >> 64;
        int64 t = val - q_estimate * mod;
        
        // Correction step: result might be in [mod, 2*mod)
        if (t >= mod) t -= mod;
        return t;
    }
};

// Single-modulus NTT, now optimized with Barrett Reduction
void ntt(int *a, int n, int mod, int type, const BarrettReducer& barrett) {
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
                // Use Barrett Reducer for the multiplication
                int64 v = barrett.reduce((__int128)w * a[i + j + len]);
                a[i + j] = (u + v) % mod;
                a[i + j + len] = (u - v + mod) % mod;
                // Use Barrett Reducer for updating the twiddle factor
                w = barrett.reduce((__int128)w * wn);
            }
        }
    }

    if (type == -1) {
        int64 inv_n = modpow(n, mod - 2, mod);
        for (int i = 0; i < n; ++i) {
            a[i] = barrett.reduce((__int128)a[i] * inv_n);
        }
    }
}

// Two-step CRT merge (no changes needed)
void crt_merge(int* r1, int* r2, int* r3, int* out, int n, int P) {
    int64 m1 = mod1;
    int64 m2 = mod2;
    int64 m3 = mod3;
    __int128 m12 = (__int128)m1 * m2;

    for (int i = 0; i < 2 * n - 1; ++i) {
        int64 a1 = r1[i];
        int64 a2 = r2[i];
        int64 a3 = r3[i];

        int64 t = ((a2 - a1) % m2 + m2) % m2;
        t = (__int128)t * inv_m1_mod_m2 % m2;
        __int128 x12 = a1 + (__int128)t * m1;

        int64 t2 = ((a3 - (int64)(x12 % m3)) % m3 + m3) % m3;
        t2 = (__int128)t2 * inv_m12_mod_m3 % m3;
        __int128 x = x12 + (__int128)t2 * (m1 * m2);

        out[i] = (int)(x % P);
    }
}

// Polynomial multiplication interface, rewritten for MPI
void poly_multiply(int *a, int *b, int *out, int n, int P) {
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int len;

    // Master Process (Rank 0)
    if (world_rank == 0) {
        len = 1;
        while (len < 2 * n) {
            len <<= 1;
        }

        // Broadcast the length to all workers
        MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);

        vector<int> a_pad(len, 0);
        vector<int> b_pad(len, 0);
        for (int i = 0; i < n; ++i) {
            a_pad[i] = a[i];
            b_pad[i] = b[i];
        }

        // Broadcast the padded polynomials to all workers
        MPI_Bcast(a_pad.data(), len, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(b_pad.data(), len, MPI_INT, 0, MPI_COMM_WORLD);

        // Receive results from workers
        vector<int> res1(len), res2(len), res3(len);
        MPI_Recv(res1.data(), len, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(res2.data(), len, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(res3.data(), len, MPI_INT, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Perform the final merge
        crt_merge(res1.data(), res2.data(), res3.data(), out, n, P);
    }
    // Worker Processes (Ranks 1, 2, 3)
    else if (world_rank >= 1 && world_rank <= 3) {
        // Receive the length from master
        MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);

        vector<int> a_pad(len);
        vector<int> b_pad(len);
        
        // Receive polynomials from master
        MPI_Bcast(a_pad.data(), len, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(b_pad.data(), len, MPI_INT, 0, MPI_COMM_WORLD);
        
        int64 current_mod;
        if (world_rank == 1) current_mod = mod1;
        if (world_rank == 2) current_mod = mod2;
        if (world_rank == 3) current_mod = mod3;

        // Perform NTT for the assigned modulus
        BarrettReducer barrett(current_mod);
        ntt(a_pad.data(), len, current_mod, 1, barrett);
        ntt(b_pad.data(), len, current_mod, 1, barrett);

        for (int i = 0; i < len; ++i) {
            a_pad[i] = barrett.reduce((__int128)a_pad[i] * b_pad[i]);
        }

        ntt(a_pad.data(), len, current_mod, -1, barrett);

        // Send the result back to the master
        MPI_Send(a_pad.data(), len, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}

// I/O helpers (no changes)
void fRead(int *a, int *b, int *n, int *p, int id) {
    ifstream fin(string("nttdata/") + to_string(id) + ".in");
    fin >> *n >> *p;
    for (int i = 0; i < *n; ++i) fin >> a[i];
    for (int i = 0; i < *n; ++i) fin >> b[i];
}

void fCheck(int *ab, int n, int id) {
    ifstream fin(string("nttdata/") + to_string(id) + ".out");
    for (int i = 0; i < 2 * n - 1; ++i) {
        int x;
        fin >> x;
        if (x != ab[i]) {
            cout << "Polynomial multiplication result is INCORRECT\n";
            return;
        }
    }
    cout << "Polynomial multiplication result is CORRECT\n";
}

void fWrite(int *ab, int n, int id) {
    ofstream fout(string("files/") + to_string(id) + ".out");
    for (int i = 0; i < 2 * n - 1; ++i) {
        fout << ab[i] << '\n';
    }
}

// Static arrays for the master process
int a[300000], b[300000], ab[600000];

int main(int argc, char *argv[]) {
    // --- MPI Initialization ---
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 4) {
        if (world_rank == 0) {
            cerr << "Error: This application requires at least 4 MPI processes (1 master + 3 workers)." << endl;
        }
        MPI_Finalize();
        return 1;
    }

    // --- Master Process Logic ---
    if (world_rank == 0) {
        cout << "Master process started. Running tests..." << endl;
        int test_begin = 0, test_end = 3;
        for (int i = test_begin; i <= test_end; ++i) {
            long double ans = 0;
            int n_, p_;
            fRead(a, b, &n_, &p_, i);
            memset(ab, 0, sizeof(ab));

            auto Start = chrono::high_resolution_clock::now();
            poly_multiply(a, b, ab, n_, p_);
            auto End = chrono::high_resolution_clock::now();
            ans += chrono::duration<double, micro>(End - Start).count();

            cout << "Test " << i << ":" << endl;
            fCheck(ab, n_, i);
            cout << "Latency for n = " << n_ << ", p = " << p_ << " : " << ans << " (us)" << endl;
            fWrite(ab, n_, i);
            cout << "---" << endl;
        }
    }
    // --- Worker Process Logic ---
    else if (world_rank >= 1 && world_rank <= 3) {
        // Workers will enter the poly_multiply function and wait for broadcasts
        // from the master. They need to loop for each test case run by the master.
        int test_begin = 0, test_end = 3;
        for (int i = test_begin; i <= test_end; ++i) {
             poly_multiply(nullptr, nullptr, nullptr, 0, 0);
        }
    }

    // --- Finalize MPI ---
    MPI_Finalize();
    return 0;
}
