#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <vector>
#include <fstream>
#include <iostream>
#include <omp.h>
using namespace std;
using int64 = long long;

// 三个模数及其原根
static const int64 mod1 = 7340033;    // 7 340 033 = 7·2^20+1, 原根 3
static const int64 mod2 = 104857601;  // 104 857 601 = 25·2^22+1, 原根 3
static const int64 mod3 = 469762049;  // 469 762 049 = 7·2^26+1, 原根 3
static const int64 G    = 3;

// CRT 逆元预计算
static const int64 inv_m1_mod_m2 = [](){ int64 a=mod1, b=mod2, u=1, v=0; while(b){ int64 t=a/b; a-=t*b; swap(a,b); u-=t*v; swap(u,v);} return (u%mod2+mod2)%mod2; }();
static const int64 inv_m12_mod_m3 = [](){ int64 m12=(int64)mod1*mod2%mod3; int64 a=m12, b=mod3, u=1, v=0; while(b){ int64 t=a/b; a-=t*b; swap(a,b); u-=t*v; swap(u,v);} return (u%mod3+mod3)%mod3; }();

int64 modpow(int64 a, int64 e, int64 m) {
    int64 r=1;
    a%=m;
    while(e) {
        if(e&1) r=(__int128)r*a%m;
        a=(__int128)a*a%m;
        e>>=1;
    }
    return r;
}

void ntt(int *a, int n, int mod, int type) {
    int L = __builtin_ctz(n);
    vector<int> rev(n);
    for(int i=1;i<n;i++) rev[i] = (rev[i>>1]>>1) | ((i&1)<<(L-1));
    for(int i=0;i<n;i++) if(i<rev[i]) swap(a[i],a[rev[i]]);

    for(int len=1;len<n;len<<=1) {
        int64 wn = modpow(G, (mod-1)/(len<<1), mod);
        if(type==-1) wn = modpow(wn, mod-2, mod);
        #pragma omp parallel for schedule(static)
        for(int i=0;i<n;i+=len<<1) {
            int64 w=1;
            for(int j=0;j<len;j++) {
                int64 u = a[i+j];
                int64 v = ( (__int128)w * a[i+j+len] ) % mod;
                a[i+j]       = (u+v)%mod;
                a[i+j+len]  = (u-v+mod)%mod;
                w = (__int128)w*wn%mod;
            }
        }
    }
    if(type==-1) {
        int64 inv_n = modpow(n, mod-2, mod);
        #pragma omp parallel for schedule(static)
        for(int i=0;i<n;i++) a[i] = (__int128)a[i]*inv_n%mod;
    }
}

// CRT 合并
void crt_merge(const vector<int>& r1, const vector<int>& r2, const vector<int>& r3,
               int* out, int n, int P) {
    int64 m1=mod1, m2=mod2, m3=mod3, m12=m1*m2;
    #pragma omp parallel for schedule(static)
    for(int i=0;i<2*n-1;i++) {
        int64 a1=r1[i], a2=r2[i], a3=r3[i];
        int64 t = ((a2 - a1)%m2 + m2)%m2;
        t = (__int128)t * inv_m1_mod_m2 % m2;
        __int128 x12 = a1 + (__int128)t*m1;
        int64 t2 = (((a3 - (int64)(x12%m3))%m3)+m3)%m3;
        t2 = (__int128)t2 * inv_m12_mod_m3 % m3;
        __int128 x = x12 + (__int128)t2*m12;
        out[i] = (int)(x % P);
    }
}

void poly_multiply(int *a, int *b, int *out, int n, int P) {
    int len=1; while(len<2*n) len<<=1;
    vector<int> padA(len), padB(len);
    copy(a, a+n, padA.begin());
    copy(b, b+n, padB.begin());
    vector<int> r1(len), r2(len), r3(len);

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            vector<int> A=padA, B=padB;
            ntt(A.data(), len, mod1, 1);
            ntt(B.data(), len, mod1, 1);
            for(int i=0;i<len;i++) A[i] = (__int128)A[i]*B[i]%mod1;
            ntt(A.data(), len, mod1, -1);
            r1 = move(A);
        }
        #pragma omp section
        {
            vector<int> A=padA, B=padB;
            ntt(A.data(), len, mod2, 1);
            ntt(B.data(), len, mod2, 1);
            for(int i=0;i<len;i++) A[i] = (__int128)A[i]*B[i]%mod2;
            ntt(A.data(), len, mod2, -1);
            r2 = move(A);
        }
        #pragma omp section
        {
            vector<int> A=padA, B=padB;
            ntt(A.data(), len, mod3, 1);
            ntt(B.data(), len, mod3, 1);
            for(int i=0;i<len;i++) A[i] = (__int128)A[i]*B[i]%mod3;
            ntt(A.data(), len, mod3, -1);
            r3 = move(A);
        }
    }
    crt_merge(r1, r2, r3, out, n, P);
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
