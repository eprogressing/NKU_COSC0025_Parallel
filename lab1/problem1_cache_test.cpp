#include <iostream>
#include <windows.h>
#include <cstring>
using namespace std;

#define N 1000

int a[N];
int b[N][N];
int result[N];
const int times = 1;

void init() {
    for (int i = 0; i < N; i++) {
        a[i] = i;
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            b[i][j] = i + j;
        }
    }
}

void common() {
    long long int begin, end, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&begin);
    
    for (int l = 0; l < times; l++) {
        memset(result, 0, sizeof(result));
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                result[i] += a[j] * b[j][i];
            }
        }
    }
    
    QueryPerformanceCounter((LARGE_INTEGER*)&end);
    cout << "common: " << (end - begin) * 1000.0 / freq / times << " ms" << endl;
}

void cache_improve() {
    long long int begin, end, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&begin);
    
    for (int l = 0; l < times; l++) {
        memset(result, 0, sizeof(result));
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                result[i] += a[j] * b[j][i];
            }
        }
    }
    
    QueryPerformanceCounter((LARGE_INTEGER*)&end);
    cout << "cache_improve: " << (end - begin) * 1000.0 / freq / times << " ms" << endl;
}

int main() {
    init();
    common();
    cache_improve();
    return 0;
}
