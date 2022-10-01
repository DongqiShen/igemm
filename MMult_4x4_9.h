#include <stdio.h>

#define A(i, j) a[(i) * lda + (j)]
#define B(i, j) b[(i) * ldb + (j)]
#define C(i, j) c[(i) * ldc + (j)]

#define Y(i) y[inc * (i)]

void AddDot(int k, double *x, int inc, double *y, double *gamma);
void AddDot4x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc);

void MMult_4x4_9(int m, int n, int k, double *a, int lda,
                                 double *b, int ldb,
                                 double *c, int ldc)
{
    for (int i = 0; i < m; i += 4) {
        for (int j = 0; j < n; j += 4) {
            AddDot4x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
        }
    }
}

void AddDot(int k, double *x, int inc, double *y, double *gamma)
{
    for (int i = 0; i < k; ++i) {
        *gamma += x[i] * Y(i);
    }
}

void AddDot4x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc)
{
    register double 
        cc_00_reg, cc_01_reg, cc_02_reg, cc_03_reg,
        cc_10_reg, cc_11_reg, cc_12_reg, cc_13_reg,
        cc_20_reg, cc_21_reg, cc_22_reg, cc_23_reg,
        cc_30_reg, cc_31_reg, cc_32_reg, cc_33_reg,

        b_p0_reg, b_p1_reg, b_p2_reg, b_p3_reg,
        a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg;

    cc_00_reg = 0.0; cc_01_reg = 0.0; cc_02_reg = 0.0; cc_03_reg = 0.0;
    cc_10_reg = 0.0; cc_11_reg = 0.0; cc_12_reg = 0.0; cc_13_reg = 0.0;
    cc_20_reg = 0.0; cc_21_reg = 0.0; cc_22_reg = 0.0; cc_23_reg = 0.0;
    cc_30_reg = 0.0; cc_31_reg = 0.0; cc_32_reg = 0.0; cc_33_reg = 0.0;

    double *a_0p_ptr, *a_1p_ptr, *a_2p_ptr, *a_3p_ptr;

    a_0p_ptr = &A(0, 0);
    a_1p_ptr = &A(1, 0);
    a_2p_ptr = &A(2, 0);
    a_3p_ptr = &A(3, 0);

    for (int i = 0; i < k; ++i) {
        b_p0_reg = B(i, 0);
        b_p1_reg = B(i, 1);
        b_p2_reg = B(i, 2);
        b_p3_reg = B(i, 3);

        a_0p_reg = *a_0p_ptr++;
        a_1p_reg = *a_1p_ptr++;
        a_2p_reg = *a_2p_ptr++;
        a_3p_reg = *a_3p_ptr++;

        //   第一列和第二列
        cc_00_reg += a_0p_reg * b_p0_reg;
        cc_01_reg += a_0p_reg * b_p1_reg;
        cc_10_reg += a_1p_reg * b_p0_reg;
        cc_11_reg += a_1p_reg * b_p1_reg;
        cc_20_reg += a_2p_reg * b_p0_reg;
        cc_21_reg += a_2p_reg * b_p1_reg;
        cc_30_reg += a_3p_reg * b_p0_reg;
        cc_31_reg += a_3p_reg * b_p1_reg;

        // 第三列和第四列
        cc_02_reg += a_0p_reg * b_p2_reg;
        cc_03_reg += a_0p_reg * b_p3_reg;
        cc_12_reg += a_1p_reg * b_p2_reg;
        cc_13_reg += a_1p_reg * b_p3_reg;
        cc_22_reg += a_2p_reg * b_p2_reg;
        cc_23_reg += a_2p_reg * b_p3_reg;
        cc_32_reg += a_3p_reg * b_p2_reg;
        cc_33_reg += a_3p_reg * b_p3_reg;
    }
    C(0, 0) = cc_00_reg; C(0, 1) = cc_01_reg; C(0, 2) = cc_02_reg; C(0, 3) = cc_03_reg;
    C(1, 0) = cc_10_reg; C(1, 1) = cc_11_reg; C(1, 2) = cc_12_reg; C(1, 3) = cc_13_reg;
    C(2, 0) = cc_20_reg; C(2, 1) = cc_21_reg; C(2, 2) = cc_22_reg; C(2, 3) = cc_23_reg;
    C(3, 0) = cc_30_reg; C(3, 1) = cc_31_reg; C(3, 2) = cc_32_reg; C(3, 3) = cc_33_reg;
}