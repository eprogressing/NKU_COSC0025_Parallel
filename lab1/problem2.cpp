#include <iostream>
#include <sys/time.h>
#include <cmath>
#include <cstdlib>
using namespace std;

#define ull unsigned long long int

int times = 10;  
void init(ull* a, ull n) 
{
    for (ull i = 0; i < n; i++)
        a[i] = i;
}

void common(ull* a, ull n) {
    struct timeval start, end;
    gettimeofday(&start, NULL);
    for (int l = 0; l < times; l++) 
    {
        ull sum = 0;
        for (ull i = 0; i < n; i++) {

            sum += a[i];
        }
        volatile ull sink = sum; 
    }
    gettimeofday(&end, NULL);
    cout << "n=" << n << "\tcommon\t\t" 
         << ((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec)) *1.0/1000/times 
         << "ms" << endl;
}

void cache_improve2(ull* a, ull n) {
    struct timeval start, end;
    gettimeofday(&start, NULL);
    for (int l = 0; l < times; l++) {
        ull sum1 = 0, sum2 = 0;
        ull i;
        for (i = 0; i < n - 1; i += 2) {
            sum1 += a[i];
            sum2 += a[i+1];
        }
        for (; i < n; i++) sum1 += a[i];
        volatile ull sink = sum1 + sum2;
    }
    gettimeofday(&end, NULL);
    cout << "n=" << n << "\tcache_improve2\t" 
         << ((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec)) *1.0/1000/times 
         << "ms" << endl;
}

void cache_improve4(ull* a, ull n) {
    struct timeval start, end;
    gettimeofday(&start, NULL);
    for (int l = 0; l < times; l++) {
        ull sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
        ull i;
        for (i = 0; i < n - 3; i += 4) {
            sum1 += a[i];
            sum2 += a[i+1];
            sum3 += a[i+2];
            sum4 += a[i+3];
        }
        for (; i < n; i++) sum1 += a[i];
        volatile ull sink = sum1 + sum2 + sum3 + sum4;
    }
    gettimeofday(&end, NULL);
    cout << "n=" << n << "\tcache_improve4\t" 
         << ((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec)) *1.0/1000/times 
         << "ms" << endl;
}

void circle(ull* a, ull n) {
    if (n == 1) return;
    for (ull i = 0; i < n / 2; i++)
        a[i] += a[n - i - 1];
    circle(a, n / 2);
}

void circle_test(ull* a, ull n) {
    struct timeval start, end;
    gettimeofday(&start, NULL);
    for (int l = 0; l < times; l++) {
        init(a, n);  
        circle(a, n);
        volatile ull sink = a[0]; 
    }
    gettimeofday(&end, NULL);
    cout << "n=" << n << "\tcircle\t\t" 
         << ((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec)) *1.0/1000/times 
         << "ms" << endl;
}

int main() {
    for (ull n = (1ULL << 10); n <= (1ULL << 30); n *= 2) {  
        ull* a = new (nothrow) ull[n];
        if (!a) {
            cerr << "Memory allocation failed for n=" << n << endl;
            continue;
        }

        init(a, n);
        
        common(a, n);
        cache_improve2(a, n);
        cache_improve4(a, n);
        circle_test(a, n);
        
        delete[] a;
        cout << "----------------------------------------" << endl;
        if (n > (1ULL << 25)) break;  
    }
    return 0;
}