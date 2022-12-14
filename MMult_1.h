#include <stdio.h>

#define A(i, j) a[(i * lda) + j]
#define B(i, j) b[(i * ldb) + j]
#define C(i, j) c[(i * ldc) + j]

#define Y(i) y[inc * (i)]

void AddDot(int k, double *x, int inc, double *y, double *gamma)
{
    for (int i = 0; i < k; ++i) {
        *gamma += x[i] * Y(i);
    }
}

void MMult_1(int m, int n, int k, double *a, int lda,
                                 double *b, int ldb,
                                 double *c, int ldc)
{
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            AddDot(k, &A(i, 0), n, &B(0, j), &C(i, j));
        }
    }
}