#include <iostream>
#include <sys/time.h>
#include <vector>
#include <algorithm>
using namespace std;

#define ull unsigned long long int 

void init(int n, vector<ull>& a, vector<vector<ull>>& b) {
    a.resize(n);
    b.resize(n, vector<ull>(n));
    for (int i = 0; i < n; i++) {
        a[i] = i;
        for (int j = 0; j < n; j++) {
            b[i][j] = i + j;
        }
    }
}

double common(int n, const vector<ull>& a, const vector<vector<ull>>& b, int times) {
    vector<ull> result(n, 0);
    struct timeval start, end;
    gettimeofday(&start, NULL);
    for (int l = 0; l < times; l++) {
        fill(result.begin(), result.end(), 0);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                result[i] += a[j] * b[j][i];
    }
    gettimeofday(&end, NULL);
    return ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000 / times;
}

double cache_improve(int n, const vector<ull>& a, const vector<vector<ull>>& b, int times) {
    vector<ull> result(n, 0);
    struct timeval start, end;
    gettimeofday(&start, NULL);
    for (int l = 0; l < times; l++) {
        fill(result.begin(), result.end(), 0);
        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++)
                result[i] += b[j][i] * a[j];
    }
    gettimeofday(&end, NULL);
    return ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000 / times;
}

int main() {
    int times = 1000;
    for (int n = 100; n <= 1000; n += 100) {
        vector<ull> a;
        vector<vector<ull>> b;
        init(n, a, b);
        double common_time = common(n, a, b, times);
        double cache_time = cache_improve(n, a, b, times);
        cout << "N = " << n << ": common = " << common_time << " ms, cache_improve = " << cache_time << " ms" << endl;
    }
    return 0;
}
