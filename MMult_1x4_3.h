#include <stdio.h>

#define A(i, j) a[(i) * lda + (j)]
#define B(i, j) b[(i) * ldb + (j)]
#define C(i, j) c[(i) * ldc + (j)]

#define Y(i) y[inc * (i)]

void AddDot(int k, double *x, int inc, double *y, double *gamma);
void AddDot1x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc);


void MMult_1x4_3(int m, int n, int k, double *a, int lda,
                                 double *b, int ldb,
                                 double *c, int ldc)
{
    for (int i = 0; i < m; i += 4) {
        for (int j = 0; j < n; ++j) {
            AddDot1x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
        }
    }
}

void AddDot(int k, double *x, int inc, double *y, double *gamma)
{
    for (int i = 0; i < k; ++i) {
        *gamma += x[i] * Y(i);
    }
}

void AddDot1x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc)
{
    AddDot(k, &A(0, 0), lda, &B(0, 0), &C(0, 0));
    AddDot(k, &A(1, 0), lda, &B(0, 0), &C(1, 0));
    AddDot(k, &A(2, 0), lda, &B(0, 0), &C(2, 0));
    AddDot(k, &A(3, 0), lda, &B(0, 0), &C(3, 0));
}