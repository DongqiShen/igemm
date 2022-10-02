#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define A(i, j) a[(i) * lda + (j)]
#define B(i, j) b[(i) * ldb + (j)]
#define C(i, j) c[(i) * ldc + (j)]

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

void print_matrix(int m, int k, double *matrix)
{
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            printf("%le ", matrix[i * k + j]);
        }
        printf("\n");
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