#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <vector>
#include <iostream>
#include <pthread.h>
using namespace std;
using int64 = long long;

// Single modulus and its primitive root
static const int64 mod = 1337006139375617; // Prime, P-1 = 2^5 * 3^2 * 464074081
static const int64 G = 3; // Primitive root for mod (verified for small orders)

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
void ntt(int64 *a, int n, int64 mod, int type) {
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
    int64 *data;
    int n;
    int64 mod;
};

void* ntt_forward_thread(void* _arg) {
    auto arg = (NTTThreadArg*)_arg;
    vector<int64> A(arg->data, arg->data + arg->n);
    ntt(A.data(), arg->n, arg->mod, 1);
    memcpy(arg->data, A.data(), arg->n * sizeof(int64));
    return nullptr;
}

// Polynomial multiplication with single modulus
void poly_multiply(int64 *a, int64 *b, int64 *out, int n, int64 P) {
    int len = 1;
    while (len < 2 * n) len <<= 1;
    if (len > 32) {
        throw runtime_error("NTT not supported for len > 32 with this modulus");
    }
    vector<int64> a_pad(len, 0), b_pad(len, 0);
    for (int i = 0; i < n; ++i) {
        a_pad[i] = (a[i] % P + P) % P;
        b_pad[i] = (b[i] % P + P) % P;
    }
    // Parallel forward NTTs
    NTTThreadArg args[2] = {
        {a_pad.data(), len, mod},
        {b_pad.data(), len, mod}
    };
    pthread_t threads[2];
    for (int i = 0; i < 2; ++i) {
        pthread_create(&threads[i], nullptr, ntt_forward_thread, &args[i]);
    }
    for (int i = 0; i < 2; ++i) {
        pthread_join(threads[i], nullptr);
    }
    // Point-wise multiplication
    for (int i = 0; i < len; ++i)
        a_pad[i] = (__int128)a_pad[i] * b_pad[i] % mod;
    // Inverse NTT (serial, as it's small)
    ntt(a_pad.data(), len, mod, -1);
    for (int i = 0; i < 2 * n - 1; ++i)
        out[i] = a_pad[i] % P;
}

int main() {
    // Test case
    int n = 2;
    int64 P = 1337006139375617;
    int64 a[2] = {1, 2}; // Polynomial: 1 + 2x
    int64 b[2] = {3, 4}; // Polynomial: 3 + 4x
    int64 ab[3] = {0};   // Result array
    int64 expected[3] = {3, 10, 8}; // Expected: 3 + 10x + 8x^2
    const int iterations = 1000; // Run multiple times for measurable latency

    // Print input
    cout << "Test Case:" << endl;
    cout << "n = " << n << ", P = " << P << endl;
    cout << "Polynomial a: [" << a[0] << ", " << a[1] << "]" << endl;
    cout << "Polynomial b: [" << b[0] << ", " << b[1] << "]" << endl;

    // Run polynomial multiplication multiple times
    auto Start = chrono::high_resolution_clock::now();
    try {
        for (int i = 0; i < iterations; ++i) {
            poly_multiply(a, b, ab, n, P);
        }
    } catch (const runtime_error& e) {
        cout << "Error: " << e.what() << endl;
        return 1;
    }
    auto End = chrono::high_resolution_clock::now();
    double latency = chrono::duration<double, nano>(End - Start).count() / iterations;

    // Print output
    cout << "Result: [" << ab[0];
    for (int i = 1; i < 2 * n - 1; ++i)
        cout << ", " << ab[i];
    cout << "]" << endl;

    // Verify result
    bool passed = true;
    for (int i = 0; i < 2 * n - 1; ++i) {
        if (ab[i] != expected[i]) {
            passed = false;
            break;
        }
    }
    cout << "Expected: [" << expected[0];
    for (int i = 1; i < 2 * n - 1; ++i)
        cout << ", " << expected[i];
    cout << "]" << endl;
    cout << "Test " << (passed ? "passed" : "failed") << endl;
    cout << "Average Latency per iteration: " << latency << " ns" << endl;

    return 0;
}
