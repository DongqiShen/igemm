#include <stdio.h>
#include "sse2neon.h"
// #include <arm_neon.h>

#define A(i, j) a[(i) * lda + (j)]
#define B(i, j) b[(i) * ldb + (j)]
#define C(i, j) c[(i) * ldc + (j)]

#define Y(i) y[inc * (i)]

void AddDot(int k, double *x, int inc, double *y, double *gamma);
void AddDot4x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc);

void MMult_4x4_10(int m, int n, int k, double *a, int lda,
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
    c_00_c_01_vreg = vdupq_n_f64(0), c_02_c_03_vreg = vdupq_n_f64(0),
    c_10_c_11_vreg = vdupq_n_f64(0), c_12_c_13_vreg = vdupq_n_f64(0),
    c_20_c_21_vreg = vdupq_n_f64(0), c_22_c_23_vreg = vdupq_n_f64(0),
    c_30_c_31_vreg = vdupq_n_f64(0), c_32_c_33_vreg = vdupq_n_f64(0);
    
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

// typedef union {
//     __m128d v;
//     double d[2];
// } v2df_t;

// void AddDot4x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc)
// {
//     v2df_t
//         c_00_c_01_vreg, c_02_c_03_vreg,
//         c_10_c_11_vreg, c_12_c_13_vreg,
//         c_20_c_21_vreg, c_22_c_23_vreg,
//         c_30_c_31_vreg, c_32_c_33_vreg,

//         b_p0_b_p1_vreg, b_p2_b_p3_vreg,
//         a_0p_vreg, a_1p_vreg, a_2p_vreg, a_3p_vreg;

//     double *a_0p_ptr, *a_1p_ptr, *a_2p_ptr, *a_3p_ptr;

//     a_0p_ptr = &A(0, 0);
//     a_1p_ptr = &A(1, 0);
//     a_2p_ptr = &A(2, 0);
//     a_3p_ptr = &A(3, 0);

//     c_00_c_01_vreg.v = _mm_setzero_pd();
//     c_02_c_03_vreg.v = _mm_setzero_pd();
//     c_10_c_11_vreg.v = _mm_setzero_pd();
//     c_12_c_13_vreg.v = _mm_setzero_pd();
//     c_20_c_21_vreg.v = _mm_setzero_pd();
//     c_22_c_23_vreg.v = _mm_setzero_pd();
//     c_30_c_31_vreg.v = _mm_setzero_pd();
//     c_32_c_33_vreg.v = _mm_setzero_pd();

//     for (int i = 0; i < k; ++i) {

//         b_p0_b_p1_vreg.v = _mm_load_pd( (double *) &B(i, 0));
//         b_p2_b_p3_vreg.v = _mm_load_pd( (double *) &B(i, 2));

//         a_0p_vreg.v = _mm_loaddup_pd((double*)a_0p_ptr++);
//         a_1p_vreg.v = _mm_loaddup_pd((double*)a_1p_ptr++);
//         a_2p_vreg.v = _mm_loaddup_pd((double*)a_2p_ptr++);
//         a_3p_vreg.v = _mm_loaddup_pd((double*)a_3p_ptr++);

//         //   ?????????????????????
//         c_00_c_01_vreg.v += a_0p_vreg.v * b_p0_b_p1_vreg.v;
//         c_10_c_11_vreg.v += a_1p_vreg.v * b_p0_b_p1_vreg.v;
//         c_20_c_21_vreg.v += a_2p_vreg.v * b_p0_b_p1_vreg.v;
//         c_30_c_31_vreg.v += a_3p_vreg.v * b_p0_b_p1_vreg.v;

//         c_02_c_03_vreg.v += a_0p_vreg.v * b_p2_b_p3_vreg.v;
//         c_12_c_13_vreg.v += a_1p_vreg.v * b_p2_b_p3_vreg.v;
//         c_22_c_23_vreg.v += a_2p_vreg.v * b_p2_b_p3_vreg.v;
//         c_32_c_33_vreg.v += a_3p_vreg.v * b_p2_b_p3_vreg.v;

//     }
//     C(0, 0) += c_00_c_01_vreg.d[0]; C(0, 1) += c_00_c_01_vreg.d[1]; 
//     C(0, 2) += c_02_c_03_vreg.d[0]; C(0, 3) += c_02_c_03_vreg.d[1];
//     C(1, 0) += c_10_c_11_vreg.d[0]; C(1, 1) += c_10_c_11_vreg.d[1];
//     C(1, 2) += c_12_c_13_vreg.d[0]; C(1, 3) += c_12_c_13_vreg.d[1];
//     C(2, 0) += c_20_c_21_vreg.d[0]; C(2, 1) += c_20_c_21_vreg.d[1];
//     C(2, 2) += c_22_c_23_vreg.d[0]; C(2, 3) += c_22_c_23_vreg.d[1];
//     C(3, 0) += c_30_c_31_vreg.d[0]; C(3, 1) += c_30_c_31_vreg.d[1];
//     C(3, 2) += c_32_c_33_vreg.d[0]; C(3, 3) += c_32_c_33_vreg.d[1];
// }