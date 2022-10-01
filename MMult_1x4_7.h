#include <stdio.h>

#define A(i, j) a[(i) * lda + (j)]
#define B(i, j) b[(i) * ldb + (j)]
#define C(i, j) c[(i) * ldc + (j)]

#define Y(i) y[inc * (i)]

void AddDot(int k, double *x, int inc, double *y, double *gamma);
void AddDot1x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc);


void MMult_1x4_7(int m, int n, int k, double *a, int lda,
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

    double *a_0i_ptr, *a_1i_ptr, *a_2i_ptr, *a_3i_ptr;
    
    cc_00_reg = 0.0;
    cc_10_reg = 0.0;
    cc_20_reg = 0.0;
    cc_30_reg = 0.0;
    b_i0_reg = 0.0;

    a_0i_ptr = &A(0, 0);
    a_1i_ptr = &A(1, 0);
    a_2i_ptr = &A(2, 0);
    a_3i_ptr = &A(3, 0);

    for (int i = 0; i < k; ++i) {
        b_i0_reg = B(i, 0);
        cc_00_reg += *a_0i_ptr++ * b_i0_reg;
        cc_10_reg += *a_1i_ptr++ * b_i0_reg;
        cc_20_reg += *a_2i_ptr++ * b_i0_reg;
        cc_30_reg += *a_3i_ptr++ * b_i0_reg;
    }
    C(0, 0) += cc_00_reg;
    C(1, 0) += cc_10_reg;
    C(2, 0) += cc_20_reg;
    C(3, 0) += cc_30_reg;
}