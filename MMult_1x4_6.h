#include <stdio.h>

#define A(i, j) a[(i) * lda + (j)]
#define B(i, j) b[(i) * ldb + (j)]
#define C(i, j) c[(i) * ldc + (j)]

#define Y(i) y[inc * (i)]

void AddDot(int k, double *x, int inc, double *y, double *gamma);
void AddDot1x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc);


void MMult_1x4_6(int m, int n, int k, double *a, int lda,
                                 double *b, int ldb,
                                 double *c, int ldc)
{
    for (int i = 0; i < m; i += 4) {
        for (int j = 0; j < n; ++j) {
            AddDot1x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
        }
    }
}

void AddDot1x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc)
{
    register double cc_00_reg, cc_10_reg, cc_20_reg, cc_30_reg, b_i0_reg;

    cc_00_reg = 0.0;
    cc_10_reg = 0.0;
    cc_20_reg = 0.0;
    cc_30_reg = 0.0;
    b_i0_reg = 0.0;

    for (int i = 0; i < k; ++i) {
        b_i0_reg = B(i, 0);
        cc_00_reg += A(0, i) * b_i0_reg;
        cc_10_reg += A(1, i) * b_i0_reg;
        cc_20_reg += A(2, i) * b_i0_reg;
        cc_30_reg += A(3, i) * b_i0_reg;
    }
    C(0, 0) += cc_00_reg;
    C(1, 0) += cc_10_reg;
    C(2, 0) += cc_20_reg;
    C(3, 0) += cc_30_reg;
}