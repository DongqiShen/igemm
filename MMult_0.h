#include "src/common.h"

// gemm C = A * sB + C
void MMult_0(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int p = 0; p < k; ++p) {
                C(i, j) = C(i, j) + A(i, p) * B(p, j);
            }
        }
    }
}