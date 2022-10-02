// #include "utils.h"
// #include "MMult_4x4_11.h"



// int main() {
//     int m = 160;
//     int n = 160;
//     int k = 160;
//     double *a = (double*)malloc(sizeof(double)*m*k);
//     double *b = (double*)malloc(sizeof(double)*k*n);
//     double *stdc = (double*)malloc(sizeof(double)*m*n);
//     double *testc = (double*)malloc(sizeof(double*)*m*n);

//     memset(stdc, 0, sizeof(double)*m*n);
//     memset(testc, 0, sizeof(double)*m*n);

//     int lda = 160;
//     int ldb = 160;
//     int ldc = 160;
//     random_matrix(m, k, a, lda);
//     random_matrix(k, n, b, ldb);

//     InnerKernel(m, n, k, a, lda, b, ldb, stdc, ldc);
//     MMult_4x4_11(m, n, k, a, lda, b, ldb, testc, ldc);
//     compare_matrix(m, n, stdc, ldc, testc, ldc);

//     free(a);
//     free(b);
//     free(stdc);
//     free(testc);
// }