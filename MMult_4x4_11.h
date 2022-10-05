#include <stdio.h>
#include <arm_neon.h>
// #include "sse2neon.h"

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

void MMult_4x4_11(int m, int n, int k, double *a, int lda, 
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
    for (int i = 0; i < m; i += 4) {
        for (int j = 0; j < n; j += 4) {
            AddDot4x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
        }
    }
}

void AddDot4x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc)
{
    float64x2_t 
        c_00_c_01_vreg, c_02_c_03_vreg,
        c_10_c_11_vreg, c_12_c_13_vreg,
        c_20_c_21_vreg, c_22_c_23_vreg,
        c_30_c_31_vreg, c_32_c_33_vreg,

        a_0p_vreg, a_1p_vreg, a_2p_vreg, a_3p_vreg,
        b_p0_b_p1_vreg, b_p2_b_p3_vreg;

    double *a_0p_ptr, *a_1p_ptr, *a_2p_ptr, *a_3p_ptr;

    a_0p_ptr = &A(0, 0);
    a_1p_ptr = &A(1, 0);
    a_2p_ptr = &A(2, 0);
    a_3p_ptr = &A(3, 0);
    
    c_00_c_01_vreg = vld1q_f64(&C(0, 0)), c_02_c_03_vreg = vld1q_f64(&C(0, 2)),
    c_10_c_11_vreg = vld1q_f64(&C(1, 0)), c_12_c_13_vreg = vld1q_f64(&C(1, 2)),
    c_20_c_21_vreg = vld1q_f64(&C(2, 0)), c_22_c_23_vreg = vld1q_f64(&C(2, 2)),
    c_30_c_31_vreg = vld1q_f64(&C(3, 0)), c_32_c_33_vreg = vld1q_f64(&C(3, 2));
    for (int p = 0; p < k; ++p) {
        b_p0_b_p1_vreg = vld1q_f64(&B(p, 0));
        b_p2_b_p3_vreg = vld1q_f64(&B(p, 2));

        a_0p_vreg = vld1q_dup_f64(a_0p_ptr++);
        a_1p_vreg = vld1q_dup_f64(a_1p_ptr++);
        a_2p_vreg = vld1q_dup_f64(a_2p_ptr++);
        a_3p_vreg = vld1q_dup_f64(a_3p_ptr++);

        // c_00_c_01_vreg = vmlaq_f64(c_00_c_01_vreg, a_0p_vreg, b_p0_b_p1_vreg);
        // c_02_c_03_vreg = vmlaq_f64(c_02_c_03_vreg, a_0p_vreg, b_p2_b_p3_vreg);
        // c_10_c_11_vreg = vmlaq_f64(c_10_c_11_vreg, a_1p_vreg, b_p0_b_p1_vreg);
        // c_12_c_13_vreg = vmlaq_f64(c_12_c_13_vreg, a_1p_vreg, b_p2_b_p3_vreg);
        // c_20_c_21_vreg = vmlaq_f64(c_20_c_21_vreg, a_2p_vreg, b_p0_b_p1_vreg);
        // c_22_c_23_vreg = vmlaq_f64(c_22_c_23_vreg, a_2p_vreg, b_p2_b_p3_vreg);
        // c_30_c_31_vreg = vmlaq_f64(c_30_c_31_vreg, a_3p_vreg, b_p0_b_p1_vreg);
        // c_32_c_33_vreg = vmlaq_f64(c_32_c_33_vreg, a_3p_vreg, b_p2_b_p3_vreg);
        c_00_c_01_vreg += a_0p_vreg * b_p0_b_p1_vreg;
        c_02_c_03_vreg += a_0p_vreg * b_p2_b_p3_vreg;
        c_10_c_11_vreg += a_1p_vreg * b_p0_b_p1_vreg;
        c_12_c_13_vreg += a_1p_vreg * b_p2_b_p3_vreg;
        c_20_c_21_vreg += a_2p_vreg * b_p0_b_p1_vreg;
        c_22_c_23_vreg += a_2p_vreg * b_p2_b_p3_vreg;
        c_30_c_31_vreg += a_3p_vreg * b_p0_b_p1_vreg;
        c_32_c_33_vreg += a_3p_vreg * b_p2_b_p3_vreg;
    }
    vst1q_f64(&C(0, 0), c_00_c_01_vreg);
    vst1q_f64(&C(0, 2), c_02_c_03_vreg);
    vst1q_f64(&C(1, 0), c_10_c_11_vreg);
    vst1q_f64(&C(1, 2), c_12_c_13_vreg);
    vst1q_f64(&C(2, 0), c_20_c_21_vreg);
    vst1q_f64(&C(2, 2), c_22_c_23_vreg);
    vst1q_f64(&C(3, 0), c_30_c_31_vreg);
    vst1q_f64(&C(3, 2), c_32_c_33_vreg);

}
