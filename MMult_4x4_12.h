#include <stdio.h>
#include "sse2neon.h"

#define A(i, j) a[(i) * lda + (j)]
#define B(i, j) b[(i) * ldb + (j)]
#define C(i, j) c[(i) * ldc + (j)]

#define Y(i) y[inc * (i)]
/* Block sizes */
#define mc 256
#define kc 128

#define min(i, j) ((i)<(j) ? (i): (j))

void AddDot(int k, double *x, int inc, double *y, double *gamma);
void AddDot4x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void InnerKernel(int m, int n, int k, double *a, int lda,
                                 double *b, int ldb,
                                 double *c, int ldc);
void PackMatrixB(int k, double *b, int ldb, double *b_to);

void MMult_4x4_12(int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
    int i, j, p, pb, ib;

    /* This time, we compute a mc x n block of C by a call to the InnerKernel */
    // A: m * k, B: k * n C: m * n
    for (i = 0; i < m; i += mc) {
        ib = min(m - i, mc);
        for (p = 0; p < k; p += kc) {
            pb = min(k - p, kc);
            InnerKernel(ib, n, pb, &A(i, p), lda, &B(p, 0), ldb, &C(i, 0), ldc);
        }
    }
}

// a: m x k, stride: lda, a是矩阵的起点
// b: k x n  stride: ldb, b是矩阵的起点
// c: m x n  stride: ldc, c是矩阵的起点
void InnerKernel(int m, int n, int k, double *a, int lda,
                                 double *b, int ldb,
                                 double *c, int ldc)
{
    double packedB[k*n];
    for (int i = 0; i < m; i += 4) {
        for (int j = 0; j < n; j += 4) {
            PackMatrixB(k, &B(0, j), ldb, &packedB[j * k]);
            AddDot4x4(k, &A(i, 0), lda, &packedB[j * k], 4, &C(i, j), ldc);
        }
    }
}

void PackMatrixB(int k, double *b, int ldb, double *b_to)
{
    for (int j = 0; j < k; ++j) {
        double *b_ij_ptr = &B(j, 0);
        *b_to++ = *b_ij_ptr;
        *b_to++ = *(b_ij_ptr + 1);
        *b_to++ = *(b_ij_ptr + 2);
        *b_to++ = *(b_ij_ptr + 3);
    }
}


typedef union {
    __m128d v;
    double d[2];
} v2df_t;

void AddDot4x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc)
{
    v2df_t
        c_00_c_01_vreg, c_02_c_03_vreg,
        c_10_c_11_vreg, c_12_c_13_vreg,
        c_20_c_21_vreg, c_22_c_23_vreg,
        c_30_c_31_vreg, c_32_c_33_vreg,

        b_p0_b_p1_vreg, b_p2_b_p3_vreg,
        a_0p_vreg, a_1p_vreg, a_2p_vreg, a_3p_vreg;

    double *a_0p_ptr, *a_1p_ptr, *a_2p_ptr, *a_3p_ptr;

    a_0p_ptr = &A(0, 0);
    a_1p_ptr = &A(1, 0);
    a_2p_ptr = &A(2, 0);
    a_3p_ptr = &A(3, 0);

    c_00_c_01_vreg.v = _mm_setzero_pd();
    c_02_c_03_vreg.v = _mm_setzero_pd();
    c_10_c_11_vreg.v = _mm_setzero_pd();
    c_12_c_13_vreg.v = _mm_setzero_pd();
    c_20_c_21_vreg.v = _mm_setzero_pd();
    c_22_c_23_vreg.v = _mm_setzero_pd();
    c_30_c_31_vreg.v = _mm_setzero_pd();
    c_32_c_33_vreg.v = _mm_setzero_pd();


    for (int i = 0; i < k; ++i) {

        b_p0_b_p1_vreg.v = _mm_load_pd( (double *) &B(i, 0));
        b_p2_b_p3_vreg.v = _mm_load_pd( (double *) &B(i, 2));

        a_0p_vreg.v = _mm_loaddup_pd((double*)a_0p_ptr++);
        a_1p_vreg.v = _mm_loaddup_pd((double*)a_1p_ptr++);
        a_2p_vreg.v = _mm_loaddup_pd((double*)a_2p_ptr++);
        a_3p_vreg.v = _mm_loaddup_pd((double*)a_3p_ptr++);

        //  第一列和第二列
        c_00_c_01_vreg.v += a_0p_vreg.v * b_p0_b_p1_vreg.v;
        c_10_c_11_vreg.v += a_1p_vreg.v * b_p0_b_p1_vreg.v;
        c_20_c_21_vreg.v += a_2p_vreg.v * b_p0_b_p1_vreg.v;
        c_30_c_31_vreg.v += a_3p_vreg.v * b_p0_b_p1_vreg.v;

        c_02_c_03_vreg.v += a_0p_vreg.v * b_p2_b_p3_vreg.v;
        c_12_c_13_vreg.v += a_1p_vreg.v * b_p2_b_p3_vreg.v;
        c_22_c_23_vreg.v += a_2p_vreg.v * b_p2_b_p3_vreg.v;
        c_32_c_33_vreg.v += a_3p_vreg.v * b_p2_b_p3_vreg.v;

    }
    C(0, 0) += c_00_c_01_vreg.d[0]; C(0, 1) += c_00_c_01_vreg.d[1]; 
    C(0, 2) += c_02_c_03_vreg.d[0]; C(0, 3) += c_02_c_03_vreg.d[1];
    C(1, 0) += c_10_c_11_vreg.d[0]; C(1, 1) += c_10_c_11_vreg.d[1];
    C(1, 2) += c_12_c_13_vreg.d[0]; C(1, 3) += c_12_c_13_vreg.d[1];
    C(2, 0) += c_20_c_21_vreg.d[0]; C(2, 1) += c_20_c_21_vreg.d[1];
    C(2, 2) += c_22_c_23_vreg.d[0]; C(2, 3) += c_22_c_23_vreg.d[1];
    C(3, 0) += c_30_c_31_vreg.d[0]; C(3, 1) += c_30_c_31_vreg.d[1];
    C(3, 2) += c_32_c_33_vreg.d[0]; C(3, 3) += c_32_c_33_vreg.d[1];
}
