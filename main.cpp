#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "utils.h"
#include "MMult_0.h"
// #include "MMult_1.h"
// #include "MMult_2.h"
// #include "MMult_1x4_3.h"
// #include "MMult_1x4_4.h"
// #include "MMult_1x4_5.h"
// #include "MMult_1x4_6.h"
// #include "MMult_1x4_7.h"
// #include "MMult_1x4_8.h"
// #include "MMult_1x4_9.h"
// #include "MMult_4x4_3.h"
// #include "MMult_4x4_4.h"
// #include "MMult_4x4_5.h"
// #include "MMult_4x4_6.h"
// #include "MMult_4x4_7.h"
// #include "MMult_4x4_8.h"
// #include "MMult_4x4_9.h"
// #include "MMult_4x4_10.h"
// #include "MMult_4x4_11.h"
// #include "MMult_4x4_12.h"
#include "MMult_4x4_13.h"
// #include "MMult_4x4_14.h"
// #include "MMult_4x4_15.h"

int m, n, k, lda, ldb, ldc;

double *a, *b, *c, *prec, *nowc;

double gflops, time_tmp, time_best, diff;

int main() {

    struct timespec start, end;

    double time_used;

    for (int i = 40; i < 500; i += 40) {
        m = i;
        n = i;
        k = i;
        
        gflops = 2.0 * m * n * k * 1.0e-9;

        lda = k;
        ldb = n;
        ldc = n;

        a = (double*)malloc(sizeof(double) * m * k);
        b = (double*)malloc(sizeof(double) * k * n);
        c = (double*)malloc(sizeof(double) * m * n);


        prec = (double*)malloc(sizeof(double) * m * n);
        nowc = (double*)malloc(sizeof(double) * m * n);

        // 随机填充矩阵
        random_matrix(m, k, a, lda);
        random_matrix(k, n, b, ldb);

        memset(prec, 0, sizeof(double) * m * n);

        copy_matrix(m, n, prec, n, nowc, n);

        // 以nowc为基准，判断矩阵运算结果是否正确
        MMult_0(m, n, k, a, lda, b, ldb, nowc, ldc);

        for (int j = 0; j < 20; ++j) {
            // 每次计算前，矩阵置0
            copy_matrix(m, n, prec, n, c, ldc);
            
            clock_gettime(CLOCK_MONOTONIC_RAW, &start);
            // 矩阵乘法放这里
            // MMult_0(m, n, k, a, lda, b, ldb, c, ldc);
            // MMult_1(m, n, k, a, lda, b, ldb, c, ldc);
            // MMult_2(m, n, k, a, lda, b, ldb, c, ldc);
            // MMult_1x4_3(m, n, k, a, lda, b, ldb, c, ldc);
            // MMult_1x4_4(m, n, k, a, lda, b, ldb, c, ldc);
            // MMult_1x4_5(m, n, k, a, lda, b, ldb, c, ldc);
            // MMult_1x4_6(m, n, k, a, lda, b, ldb, c, ldc);
            // MMult_1x4_7(m, n, k, a, lda, b, ldb, c, ldc);
            // MMult_1x4_8(m, n, k, a, lda, b, ldb, c, ldc);
            // MMult_1x4_9(m, n, k, a, lda, b, ldb, c, ldc);
            // MMult_4x4_3(m, n, k, a, lda, b, ldb, c, ldc);
            // MMult_4x4_4(m, n, k, a, lda, b, ldb, c, ldc);
            // MMult_4x4_5(m, n, k, a, lda, b, ldb, c, ldc);
            // MMult_4x4_6(m, n, k, a, lda, b, ldb, c, ldc);
            // MMult_4x4_7(m, n, k, a, lda, b, ldb, c, ldc);
            // MMult_4x4_8(m, n, k, a, lda, b, ldb, c, ldc);
            // MMult_4x4_9(m, n, k, a, lda, b, ldb, c, ldc);
            // MMult_4x4_10(m, n, k, a, lda, b, ldb, c, ldc);
            // MMult_4x4_11(m, n, k, a, lda, b, ldb, c, ldc);
            // MMult_4x4_12(m, n, k, a, lda, b, ldb, c, ldc);
            MMult_4x4_13(m, n, k, a, lda, b, ldb, c, ldc);
            // MMult_4x4_14(m, n, k, a, lda, b, ldb, c, ldc);
            // MMult_4x4_15(m, n, k, a, lda, b, ldb, c, ldc);

            clock_gettime(CLOCK_MONOTONIC_RAW, &end);

            time_tmp = get_time(&start, &end);

            if (j == 0) {
                time_best = time_tmp;
            } else {
                time_best = fmin(time_best, time_tmp);
            }
        }
        diff = compare_matrix(m, n, c, ldc, nowc, ldc);

        if (diff > 0.5f || diff < -0.5f) {
            exit(0);
        }
        
        printf("%d %le %le\n", i, gflops / time_best, diff);

        fflush(stdout);

        free(a);
        free(b);
        free(c);
        free(prec);
        free(nowc);
    }
    printf("\n");
    fflush(stdout);
    return 0;
}