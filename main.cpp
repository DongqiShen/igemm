#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "matrix_multiply_origin.h"
// #include "MMult1.h"
// #include "MMult2.h"
#include "MMult_1x4_3.h"

void random_matrix(int m, int n, double *a, int lda)
{
    double drand48();

    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            A(i, j) = (double)rand() / (double)RAND_MAX;
        }
    }
}

void copy_matrix(int m, int n, double *a, int lda, double *b, int ldb)
{
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            B(i, j) = A(i, j);
        }
    }
}

double compare_matrix(int m, int n, double *a, int lda, double *b, int ldb)
{
    double diff;
    double max_diff = 0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            diff = fabs(A(i, j) - B(i, j));
            max_diff = fmax(diff, max_diff);
            if (max_diff > 0.5f) {
                printf("\n error at position (%d, %d), diff %f", i, j, max_diff);
            }
        }
    }
    return max_diff;
}

static double get_time(struct timespec *start, struct timespec *end)
{
    return (end->tv_sec - start->tv_sec) + (end->tv_nsec - start->tv_nsec) * 1e-9;
}

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
        MatrixMultiply(m, n, k, a, lda, b, ldb, nowc, ldc);

        for (int j = 0; j < 20; ++j) {
            // 每次计算前，矩阵置0
            copy_matrix(m, n, prec, n, c, ldc);
            
            clock_gettime(CLOCK_MONOTONIC_RAW, &start);
            // 矩阵乘法放这里
            // MatrixMultiply(m, n, k, a, lda, b, ldb, c, ldc);
            // MMult1(m, n, k, a, lda, b, ldb, c, ldc);
            // MMult2(m, n, k, a, lda, b, ldb, c, ldc);
            MMult_1x4_3(m, n, k, a, lda, b, ldb, c, ldc);

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