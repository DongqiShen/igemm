# Optimize GEMM step by step

## GEMM intro
GEMM是General matrix multiply的缩写，也即通用矩阵乘法的简称。更广泛得说，GEMM是BLAS(Basic Linear Algebra Subprograms)的一个子集，后者代表了通用线性代数计算，比方说向量加法，标量乘法，点乘，矩阵乘法等计算的集合。在不同的硬件上，有许多针对特定硬件进行优化的库，根据硬件的特性，优化计算性能。在本章中，主要探讨了在cpu下，如何一步步优化gemm的计算效率。

矩阵乘法可以表示为``C = A x B``，他们的形状分别为``C: [m, n]``, ``A: [m, k]``和``B: [k, n]``。其最直观的计算代码如下
```c
for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
        for (int p = 0; p < k; ++p) {
            C(i, j) = C(i, j) + A(i, p) * B(p, j);
        }
    }
}
```
在每次计算中，包括了两次浮点运算(加法和乘法)，因此其计算复杂度为``2mnk``。在接下来一步步优化的过程中，充分利用了硬件的优势，这里需要一点计算机体系结构的知识，使得它的计算效率得到极大提升。

为了更加直观地表示，以及专注于计算的过程的优化，首先在这里定义一些规则。
首先是矩阵的表示
```c++
double *a[m*k];
double *b[k*n];
double *c[m*n];
```
用一维的数组表示一个矩阵，比较符合数据在内存中的真实布局。并用下面的宏针对二维的位置进行访问。需要注意的是，这里采用了行主序的内存布局。其中lda表示*leading dimension of two-dimensional array a*，即矩阵a的列数，其他两个矩阵同理。
```c++
#define A(i, j) a[(i * lda) + j]
#define B(i, j) b[(i * ldb) + j]
#define C(i, j) c[(i * ldc) + j]
```
通过宏访问的时候，实际上是在数组a起始地址加上一个偏移。这一点在后续的优化中会有一定的体现。
## Result

### MMult0.c
这是最原始的一个版本，它的C代码如上一节所示。
|矩阵大小|gflops/sec|diff|
|---|---|---|
|40 | 6.141563e-01| 0.000000e+00|
|80 | 5.996341e-01| 0.000000e+00|
|120| 5.949347e-01| 0.000000e+00|
|160| 5.867897e-01| 0.000000e+00|
|200| 5.897105e-01| 0.000000e+00|
|240| 5.888551e-01| 0.000000e+00|
|280| 5.878294e-01| 0.000000e+00|
|320| 5.834000e-01| 0.000000e+00|
|360| 5.859244e-01| 0.000000e+00|
|400| 5.869953e-01| 0.000000e+00|
|440| 5.847044e-01| 0.000000e+00|
|480| 5.841649e-01| 0.000000e+00|

### MMult1.c
在第一次优化，首先把最内层循环写成一个子程序。理论上存在一定的优化空间。在最原始的版本中，每次对元素的访问，都是**相对数组起始位置做偏移**，而在函数``AddDot``中，其参数``*x``和``*y``已经**相对原始位置做了一定的偏移**。根据计算机的局部性原理，以及访问数组访问过程中的下标的计算，理论上是存在一定的优化空间。从结果可以看出，在小矩阵的计算上，提升较为明显，最多为``10%``左右，但是在大矩阵的计算上，几乎是持平的。
```c++
#define Y(i) y[inc * (i)]

void AddDot(int k, double *x, int inc, double *y, double *gamma)
{
    for (int i = 0; i < k; ++i) {
        *gamma += x[i] * Y(i);
    }
}

void MMult_1(int m, int n, int k, double *a, int lda,
                                  double *b, int ldb,
                                  double *c, int ldc)
{
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            AddDot(k, &A(i, 0), n, &B(0, j), &C(i, j));
        }
    }
}
```
|矩阵大小|gflops/sec|diff|
|---|---|---|
|40 |7.063706e-01 |0.000000e+00|
|80 |6.398668e-01 |0.000000e+00|
|120| 6.211964e-01| 0.000000e+00|
|160| 6.087576e-01| 0.000000e+00|
|200| 6.046901e-01| 0.000000e+00|
|240| 6.028132e-01| 0.000000e+00|
|280| 5.998736e-01| 0.000000e+00|
|320| 5.843010e-01| 0.000000e+00|
|360| 5.959986e-01| 0.000000e+00|
|400| 5.948869e-01| 0.000000e+00|
|440| 5.930018e-01| 0.000000e+00|
|480| 5.902639e-01| 0.000000e+00|

### MMult2.c
在**MMult_1**的基础上做了简单的修改，每次循环计算四行。这里的优化依据是提高**cache命中率**。相连的四行在内存中排布是连在一起的。不过从结果来看，并没有提升。
```c++
void MMult_2(int m, int n, int k, double *a, int lda,
                                 double *b, int ldb,
                                 double *c, int ldc)
{
    for (int i = 0; i < m; i += 4) {
        for (int j = 0; j < n; ++j) {
            AddDot(k, &A(i, 0), n, &B(0, j), &C(i, j));
            AddDot(k, &A(i + 1, 0), n, &B(0, j), &C(i + 1, j));
            AddDot(k, &A(i + 2, 0), n, &B(0, j), &C(i + 2, j));
            AddDot(k, &A(i + 3, 0), n, &B(0, j), &C(i + 3, j));
        }
    }
}

void AddDot(int k, double *x, int inc, double *y, double *gamma)
{
    for (int i = 0; i < k; ++i) {
        *gamma += x[i] * Y(i);
    }
}
```

|矩阵大小|gflops/sec|diff|
|---|---|---|
|40 |6.994536e-01 |0.000000e+00|
|80 |6.418723e-01 |0.000000e+00|
|120| 6.195769e-01| 0.000000e+00|
|160| 6.095786e-01| 0.000000e+00|
|200| 6.047892e-01| 0.000000e+00|
|240| 6.031545e-01| 0.000000e+00|
|280| 6.005458e-01| 0.000000e+00|
|320| 5.917278e-01| 0.000000e+00|
|360| 5.961168e-01| 0.000000e+00|
|400| 5.922830e-01| 0.000000e+00|
|440| 5.918232e-01| 0.000000e+00|
|480| 5.893593e-01| 0.000000e+00|

### MMult_1x4_3.c
从本次优化开始，一次计算四个元素。最内层循环写成一个子函数，一次计算四行一列，并放在一个子函数里面。在**MMult_2**中，每次计算指针的地址都是相对于指针的起点，但是在这里只有第一次是相对于指针的起点，在子函数中，每次的地址偏移都是相对于偏移后的指针，分别是相差[0, 1, 2, 3]行个地址偏移。不过，结果显示，依然没有性能提升。
```c++
void MMult_1x4_3(int m, int n, int k, double *a, int lda,
                                 double *b, int ldb,
                                 double *c, int ldc)
{
    for (int i = 0; i < m; i += 4) {
        for (int j = 0; j < n; ++j) {
            AddDot1x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
        }
    }
}
void AddDot(int k, double *x, int inc, double *y, double *gamma)
{
    for (int i = 0; i < k; ++i) {
        *gamma += x[i] * Y(i);
    }
}
void AddDot1x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc)
{
    AddDot(k, &A(0, 0), lda, &B(0, 0), &C(0, 0));
    AddDot(k, &A(1, 0), lda, &B(0, 0), &C(1, 0));
    AddDot(k, &A(2, 0), lda, &B(0, 0), &C(2, 0));
    AddDot(k, &A(3, 0), lda, &B(0, 0), &C(3, 0));
}
```
|矩阵大小|gflops/sec|diff|
|---|---|---|
|40 |7.078315e-01 |0.000000e+00|
|80 |6.408010e-01 |0.000000e+00|
|120| 6.190682e-01| 0.000000e+00|
|160| 6.094369e-01| 0.000000e+00|
|200| 6.047806e-01| 0.000000e+00|
|240| 6.017040e-01| 0.000000e+00|
|280| 5.989342e-01| 0.000000e+00|
|320| 5.915427e-01| 0.000000e+00|
|360| 5.888644e-01| 0.000000e+00|
|400| 5.919400e-01| 0.000000e+00|
|440| 5.904501e-01| 0.000000e+00|
|480| 5.884145e-01| 0.000000e+00|

### MMult_1x4_4.c
上面的```inline```版，跟猜测的一样，没有性能提升。
```c++
void MMult_1x4_4(int m, int n, int k, double *a, int lda,
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
    for (int i = 0; i < k; ++i) {
        C(0, 0) += A(0, i) * B(i, 0);
    }
    for (int i = 0; i < k; ++i) {
        C(1, 0) += A(1, i) * B(i, 0);
    }
    for (int i = 0; i < k; ++i) {
        C(2, 0) += A(2, i) * B(i, 0);
    }
    for (int i = 0; i < k; ++i) {
        C(3, 0) += A(3, i) * B(i, 0);
    }
}
```
|矩阵大小|gflops/sec|diff|
|---|---|---|
|40 |6.983393e-01 |0.000000e+00|
|80 |6.345799e-01 |0.000000e+00|
|120| 6.160519e-01| 0.000000e+00|
|160| 6.018090e-01| 0.000000e+00|
|200| 5.949965e-01| 0.000000e+00|
|240| 5.917903e-01| 0.000000e+00|
|280| 5.866356e-01| 0.000000e+00|
|320| 5.824141e-01| 0.000000e+00|
|360| 5.828859e-01| 0.000000e+00|
|400| 5.821491e-01| 0.000000e+00|
|440| 5.788143e-01| 0.000000e+00|
|480| 5.740661e-01| 0.000000e+00|

### MMult_1x4_5.c
将四个for循环合并。这个结果令人惊讶，性能翻了一倍多。这里的提升主要为两个方面：
1. 在每8次浮点运算中，变量``i``只需要变化一次。
2. ``B(i, 0)``只需要从内存中取1次，而不是四次。（这个优化只有在矩阵不能放入``L2 cache``时才有效）
```C++
void MMult_1x4_5(int m, int n, int k, double *a, int lda,
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
    for (int i = 0; i < k; ++i) {
        C(0, 0) += A(0, i) * B(i, 0);
        C(1, 0) += A(1, i) * B(i, 0);
        C(2, 0) += A(2, i) * B(i, 0);
        C(3, 0) += A(3, i) * B(i, 0);
    }
}
```
|矩阵大小|gflops/sec|diff|
|---|---|---|
|40 |1.496341e+00 |0.000000e+00|
|80 |1.520793e+00 |0.000000e+00|
|120| 1.530106e+00| 0.000000e+00|
|160| 1.498091e+00| 0.000000e+00|
|200| 1.511394e+00| 0.000000e+00|
|240| 1.503058e+00| 0.000000e+00|
|280| 1.497259e+00| 0.000000e+00|
|320| 1.344291e+00| 0.000000e+00|
|360| 1.494598e+00| 0.000000e+00|
|400| 1.488700e+00| 0.000000e+00|
|440| 1.492487e+00| 0.000000e+00|
|480| 1.387479e+00| 0.000000e+00|

### MMult_1x4_6.c 
使用了关键字**register**。注意，在**c++17**中，这个关键字已经废弃。这里使用的是**c++14**。通过显示得将变量放在寄存器中，提升计算效率。总共使用了5个寄存器，其中4个寄存器分别放累加值，另一个存放共用的``B(i, 0)`。结果表明，性能提升也是相对比较明显的。不过考虑到这个关键词已经废弃，在c代码中显式操作寄存器，有点不太合理。根据[官方文档](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4340.html)说明，在编译器的优化过程中，**register**很多时候都没有起作用。
```c++
void MMult_1x4_6(int m, int n, int k, double *a, int lda,
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
    cc_00_reg = 0.0;
    cc_10_reg = 0.0;
    cc_20_reg = 0.0;
    cc_30_reg = 0.0;
    b_i0_reg = 0.0;
    for (int i = 0; i < k; ++i) {
        b_i0_reg = B(i, 0);
        cc_00_reg += A(0, i) * b_i0_reg;
        cc_10_reg += A(1, i) * b_i0_reg;
        cc_20_reg += A(2, i) * b_i0_reg;
        cc_30_reg += A(3, i) * b_i0_reg;
    }
    C(0, 0) += cc_00_reg;
    C(1, 0) += cc_10_reg;
    C(2, 0) += cc_20_reg;
    C(3, 0) += cc_30_reg;
}
```
|矩阵大小|gflops/sec|diff|
|---|---|---|
|40 |1.945555e+00 |0.000000e+00|
|80 |1.970967e+00 |0.000000e+00|
|120| 2.009400e+00| 0.000000e+00|
|160| 1.997967e+00| 0.000000e+00|
|200| 1.966327e+00| 0.000000e+00|
|240| 1.964840e+00| 0.000000e+00|
|280| 1.990547e+00| 0.000000e+00|
|320| 1.642673e+00| 0.000000e+00|
|360| 1.980038e+00| 0.000000e+00|
|400| 1.993397e+00| 0.000000e+00|
|440| 1.984006e+00| 0.000000e+00|
|480| 1.772945e+00| 0.000000e+00|

### MMult_1x4_7.c
对比**MMult_1x4_6**中的实现，最主要的区别是在访问矩阵a每一行的数据时，通过使用指针累加的方式实现，而不是每次访问都是针对每四行的行首进行偏移。在本次优化中，提升也是相对比较可观的。
```c++
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
```
|矩阵大小|gflops/sec|diff|
|---|---|---|
|40 |2.128898e+00 |0.000000e+00|
|80 |2.209082e+00 |0.000000e+00|
|120| 2.218999e+00| 0.000000e+00|
|160| 2.190008e+00| 0.000000e+00|
|200| 2.142905e+00| 0.000000e+00|
|240| 2.145619e+00| 0.000000e+00|
|280| 2.152346e+00| 0.000000e+00|
|320| 1.775540e+00| 0.000000e+00|
|360| 2.137807e+00| 0.000000e+00|
|400| 2.141716e+00| 0.000000e+00|
|440| 2.141772e+00| 0.000000e+00|
|480| 1.904908e+00| 0.000000e+00|

### MMult_1x4_8.c
进一步展开，在一次循环中计算4列，性能有一定提升，但是并不明显。
```C++
void MMult_1x4_8(int m, int n, int k, double *a, int lda,
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

    for (int i = 0; i < k; i += 4) {
        b_i0_reg = B(i, 0);

        cc_00_reg += *a_0i_ptr++ * b_i0_reg;
        cc_10_reg += *a_1i_ptr++ * b_i0_reg;
        cc_20_reg += *a_2i_ptr++ * b_i0_reg;
        cc_30_reg += *a_3i_ptr++ * b_i0_reg;

        b_i0_reg = B(i + 1, 0);

        cc_00_reg += *a_0i_ptr++ * b_i0_reg;
        cc_10_reg += *a_1i_ptr++ * b_i0_reg;
        cc_20_reg += *a_2i_ptr++ * b_i0_reg;
        cc_30_reg += *a_3i_ptr++ * b_i0_reg;

        b_i0_reg = B(i + 2, 0);

        cc_00_reg += *a_0i_ptr++ * b_i0_reg;
        cc_10_reg += *a_1i_ptr++ * b_i0_reg;
        cc_20_reg += *a_2i_ptr++ * b_i0_reg;
        cc_30_reg += *a_3i_ptr++ * b_i0_reg;

        b_i0_reg = B(i + 3, 0);

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
```
|矩阵大小|gflops/sec|diff|
|---|---|---|
|40 |2.263804e+00 |0.000000e+00|
|80 |2.268624e+00 |0.000000e+00|
|120| 2.293615e+00| 0.000000e+00|
|160| 2.256179e+00| 0.000000e+00|
|200| 2.238937e+00| 0.000000e+00|
|240| 2.237738e+00| 0.000000e+00|
|280| 2.256632e+00| 0.000000e+00|
|320| 1.936512e+00| 0.000000e+00|
|360| 2.270535e+00| 0.000000e+00|
|400| 2.267965e+00| 0.000000e+00|
|440| 2.276264e+00| 0.000000e+00|
|480| 2.071734e+00| 0.000000e+00|

### MMult_1x4_9.c
这个版本改动比较小，只是指针``a_0i_ptr``不采用自累加的方式，访问元素的时候通过偏移来实现，在一次循环中，其值的变动从4次变为1次。不太清楚这个优化的依据，实测结果也显示没有优化。
```c++
void MMult_1x4_9(int m, int n, int k, double *a, int lda,
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

    for (int i = 0; i < k; i += 4) {
        
        b_i0_reg = B(i, 0);

        cc_00_reg += *a_0i_ptr * b_i0_reg;
        cc_10_reg += *a_1i_ptr * b_i0_reg;
        cc_20_reg += *a_2i_ptr * b_i0_reg;
        cc_30_reg += *a_3i_ptr * b_i0_reg;

        b_i0_reg = B(i + 1, 0);

        cc_00_reg += *(a_0i_ptr + 1) * b_i0_reg;
        cc_10_reg += *(a_1i_ptr + 1) * b_i0_reg;
        cc_20_reg += *(a_2i_ptr + 1) * b_i0_reg;
        cc_30_reg += *(a_3i_ptr + 1) * b_i0_reg;

        b_i0_reg = B(i + 2, 0);

        cc_00_reg += *(a_0i_ptr + 2) * b_i0_reg;
        cc_10_reg += *(a_1i_ptr + 2) * b_i0_reg;
        cc_20_reg += *(a_2i_ptr + 2) * b_i0_reg;
        cc_30_reg += *(a_3i_ptr + 2) * b_i0_reg;

        b_i0_reg = B(i + 3, 0);

        cc_00_reg += *(a_0i_ptr + 3) * b_i0_reg;
        cc_10_reg += *(a_1i_ptr + 3) * b_i0_reg;
        cc_20_reg += *(a_2i_ptr + 3) * b_i0_reg;
        cc_30_reg += *(a_3i_ptr + 3) * b_i0_reg;

        a_0i_ptr += 4;
        a_1i_ptr += 4;
        a_2i_ptr += 4;
        a_3i_ptr += 4;
    }
    C(0, 0) += cc_00_reg;
    C(1, 0) += cc_10_reg;
    C(2, 0) += cc_20_reg;
    C(3, 0) += cc_30_reg;
}
```
|矩阵大小|gflops/sec|diff|
|---|---|---|
|40 |2.247270e+00 |0.000000e+00|
|80 |2.284863e+00 |0.000000e+00|
|120| 2.283260e+00| 0.000000e+00|
|160| 2.236037e+00| 0.000000e+00|
|200| 2.234299e+00| 0.000000e+00|
|240| 2.239218e+00| 0.000000e+00|
|280| 2.260592e+00| 0.000000e+00|
|320| 1.899831e+00| 0.000000e+00|
|360| 2.259968e+00| 0.000000e+00|
|400| 2.255563e+00| 0.000000e+00|
|440| 2.235402e+00| 0.000000e+00|
|480| 2.010742e+00| 0.000000e+00|


### MMult_1x4 总结
在一步步的优化过程中可以发现一共经历了四次比较明显的性能提升。
1. 第一次是在**MMult_1x4_5**中分块思想的运用，在一次循环中计算四行的值，性能几乎翻了三倍，这也是提升最大的优化。
2. 第二次是紧接着在**MMult_1x4_6**中，使用了**register**寄存器，提升也较为明显，不过考虑到该关键字已经废弃，后续的替代方案值得想一下。
3. 第三次是在**MMult_1x4_7**中，使用了指针累加的方式访问每一行的元素，而不是每次都在行首指针加上一个offset。
4. 第四次是在**Mult_1x4_8**中，增加了展开，减少了循环的次数，增加每次循环的计算量，这种置换能带来一些性能上的提升，其本质也和第一次一样，分块。

对比原始版本，优化后的性能翻了三倍多，提升还是非常可观的。

### MMult_4x4_3.c
从本节开始，每次计算矩阵C中4x4共16个元素的值。同通俗得说，矩阵外面两层循环每次均+4，也即最内层的子程序内，要完成16个值的计算。首先是最原始的版本，和**MMult_0**一样，几乎谈不上有任何的优化。

```c++
void MMult_4x4_3(int m, int n, int k, double *a, int lda,
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
    AddDot(k, &A(0, 0), lda, &B(0, 0), &C(0, 0));
    AddDot(k, &A(0, 0), lda, &B(0, 1), &C(0, 1));
    AddDot(k, &A(0, 0), lda, &B(0, 2), &C(0, 2));
    AddDot(k, &A(0, 0), lda, &B(0, 3), &C(0, 3));

    AddDot(k, &A(1, 0), lda, &B(0, 0), &C(1, 0));
    AddDot(k, &A(1, 0), lda, &B(0, 1), &C(1, 1));
    AddDot(k, &A(1, 0), lda, &B(0, 2), &C(1, 2));
    AddDot(k, &A(1, 0), lda, &B(0, 3), &C(1, 3));

    AddDot(k, &A(2, 0), lda, &B(0, 0), &C(2, 0));
    AddDot(k, &A(2, 0), lda, &B(0, 1), &C(2, 1));
    AddDot(k, &A(2, 0), lda, &B(0, 2), &C(2, 2));
    AddDot(k, &A(2, 0), lda, &B(0, 3), &C(2, 3));

    AddDot(k, &A(3, 0), lda, &B(0, 0), &C(3, 0));
    AddDot(k, &A(3, 0), lda, &B(0, 1), &C(3, 1));
    AddDot(k, &A(3, 0), lda, &B(0, 2), &C(3, 2));
    AddDot(k, &A(3, 0), lda, &B(0, 3), &C(3, 3));
}
```
|矩阵大小|gflops/sec|diff|
|---|---|---|
|40 |7.094675e-01 |0.000000e+00|
|80 |6.399332e-01 |0.000000e+00|
|120| 6.172302e-01| 0.000000e+00|
|160| 6.114954e-01| 0.000000e+00|
|200| 6.046749e-01| 0.000000e+00|
|240| 6.030005e-01| 0.000000e+00|
|280| 6.008351e-01| 0.000000e+00|
|320| 5.928836e-01| 0.000000e+00|
|360| 5.957017e-01| 0.000000e+00|
|400| 5.946228e-01| 0.000000e+00|
|440| 5.927450e-01| 0.000000e+00|
|480| 5.883822e-01| 0.000000e+00|

### MMult_4x4_4.c
对应于**MMult_1x4_4.c*，一次计算4x4的块，使用inline，而不是上述的函数调用，性能上于前者基本一致，也谈不上有什么优化。
```c++
void MMult_4x4_4(int m, int n, int k, double *a, int lda,
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
    for (int i = 0; i < k; ++i) {
        C(0, 0) += A(0, i) * B(i, 0);
    }
    for (int i = 0; i < k; ++i) {
        C(0, 1) += A(0, i) * B(i, 1);
    }
    for (int i = 0; i < k; ++i) {
        C(0, 2) += A(0, i) * B(i, 2);
    }
    for (int i = 0; i < k; ++i) {
        C(0, 3) += A(0, i) * B(i, 3);
    }
    for (int i = 0; i < k; ++i) {
        C(1, 0) += A(1, i) * B(i, 0);
    }
    for (int i = 0; i < k; ++i) {
        C(1, 1) += A(1, i) * B(i, 1);
    }
    for (int i = 0; i < k; ++i) {
        C(1, 2) += A(1, i) * B(i, 2);
    }
    for (int i = 0; i < k; ++i) {
        C(1, 3) += A(1, i) * B(i, 3);
    }
    for (int i = 0; i < k; ++i) {
        C(2, 0) += A(2, i) * B(i, 0);
    }
    for (int i = 0; i < k; ++i) {
        C(2, 1) += A(2, i) * B(i, 1);
    }
    for (int i = 0; i < k; ++i) {
        C(2, 2) += A(2, i) * B(i, 2);
    }
    for (int i = 0; i < k; ++i) {
        C(2, 3) += A(2, i) * B(i, 3);
    }
    for (int i = 0; i < k; ++i) {
        C(3, 0) += A(3, i) * B(i, 0);
    }
    for (int i = 0; i < k; ++i) {
        C(3, 1) += A(3, i) * B(i, 1);
    }
    for (int i = 0; i < k; ++i) {
        C(3, 2) += A(3, i) * B(i, 2);
    }
    for (int i = 0; i < k; ++i) {
        C(3, 3) += A(3, i) * B(i, 3);
    }
}
```
|矩阵大小|gflops/sec|diff|
|---|---|---|
|40 |7.028140e-01 |0.000000e+00|
|80 |6.373443e-01 |0.000000e+00|
|120| 6.178600e-01| 0.000000e+00|
|160| 6.042282e-01| 0.000000e+00|
|200| 6.045540e-01| 0.000000e+00|
|240| 5.987526e-01| 0.000000e+00|
|280| 5.948014e-01| 0.000000e+00|
|320| 5.890729e-01| 0.000000e+00|
|360| 5.957199e-01| 0.000000e+00|
|400| 5.902067e-01| 0.000000e+00|
|440| 5.886776e-01| 0.000000e+00|
|480| 5.882766e-01| 0.000000e+00|

### MMult_4x4_5.c
参照**MMult_1x4_5**的结果，同样的优化放在这里，应该有很大的性能提升。即将16个计算放在一个循环里面，不如所料，性能提升明显。并且相对于**1x4**版本的结果，效果更好。通过这两个拥有同样优化思想的版本的对比，也可以看出，减少循环次数，增加每次循环的计算强度，性能提升相对明显。
```c++
void MMult_4x4_5(int m, int n, int k, double *a, int lda,
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
    for (int i = 0; i < k; ++i) {
        C(0, 0) += A(0, i) * B(i, 0);
        C(0, 1) += A(0, i) * B(i, 1);
        C(0, 2) += A(0, i) * B(i, 2);
        C(0, 3) += A(0, i) * B(i, 3);
        C(1, 0) += A(1, i) * B(i, 0);
        C(1, 1) += A(1, i) * B(i, 1);
        C(1, 2) += A(1, i) * B(i, 2);
        C(1, 3) += A(1, i) * B(i, 3);
        C(2, 0) += A(2, i) * B(i, 0);
        C(2, 1) += A(2, i) * B(i, 1);
        C(2, 2) += A(2, i) * B(i, 2);
        C(2, 3) += A(2, i) * B(i, 3);
        C(3, 0) += A(3, i) * B(i, 0);
        C(3, 1) += A(3, i) * B(i, 1);
        C(3, 2) += A(3, i) * B(i, 2);
        C(3, 3) += A(3, i) * B(i, 3);
    }
}
```
|矩阵大小|gflops/sec|diff|
|---|---|---|
|40 |1.655180e+00 |0.000000e+00|
|80 |1.663239e+00 |0.000000e+00|
|120| 1.651087e+00| 0.000000e+00|
|160| 1.628952e+00| 0.000000e+00|
|200| 1.624785e+00| 0.000000e+00|
|240| 1.633439e+00| 0.000000e+00|
|280| 1.636509e+00| 0.000000e+00|
|320| 1.608334e+00| 0.000000e+00|
|360| 1.648612e+00| 0.000000e+00|
|400| 1.645963e+00| 0.000000e+00|
|440| 1.645372e+00| 0.000000e+00|
|480| 1.623171e+00| 0.000000e+00|

### MMult_4x4_6.c
在**MMult_4x4_5**的基础上使用了**register**关键字，性能提升也相对比较明显。
```c++
void AddDot4x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc)
{
    register double 
        cc_00_reg, cc_01_reg, cc_02_reg, cc_03_reg,
        cc_10_reg, cc_11_reg, cc_12_reg, cc_13_reg,
        cc_20_reg, cc_21_reg, cc_22_reg, cc_23_reg,
        cc_30_reg, cc_31_reg, cc_32_reg, cc_33_reg,

        b_p0_reg, b_p1_reg, b_p2_reg, b_p3_reg;

    cc_00_reg = 0.0; cc_01_reg = 0.0; cc_02_reg = 0.0; cc_03_reg = 0.0;
    cc_10_reg = 0.0; cc_11_reg = 0.0; cc_12_reg = 0.0; cc_13_reg = 0.0;
    cc_20_reg = 0.0; cc_21_reg = 0.0; cc_22_reg = 0.0; cc_23_reg = 0.0;
    cc_30_reg = 0.0; cc_31_reg = 0.0; cc_32_reg = 0.0; cc_33_reg = 0.0;

    for (int i = 0; i < k; ++i) {
        b_p0_reg = B(i, 0);
        b_p1_reg = B(i, 1);
        b_p2_reg = B(i, 2);
        b_p3_reg = B(i, 3);

        cc_00_reg += A(0, i) * b_p0_reg;
        cc_01_reg += A(0, i) * b_p1_reg;
        cc_02_reg += A(0, i) * b_p2_reg;
        cc_03_reg += A(0, i) * b_p3_reg;
        
        cc_10_reg += A(1, i) * b_p0_reg;
        cc_11_reg += A(1, i) * b_p1_reg;
        cc_12_reg += A(1, i) * b_p2_reg;
        cc_13_reg += A(1, i) * b_p3_reg;
        
        cc_20_reg += A(2, i) * b_p0_reg;
        cc_21_reg += A(2, i) * b_p1_reg;
        cc_22_reg += A(2, i) * b_p2_reg;
        cc_23_reg += A(2, i) * b_p3_reg;
        
        cc_30_reg += A(3, i) * b_p0_reg;
        cc_31_reg += A(3, i) * b_p1_reg;
        cc_32_reg += A(3, i) * b_p2_reg;
        cc_33_reg += A(3, i) * b_p3_reg;
    }
    C(0, 0) = cc_00_reg; C(0, 1) = cc_01_reg; C(0, 2) = cc_02_reg; C(0, 3) = cc_03_reg;
    C(1, 0) = cc_10_reg; C(1, 1) = cc_11_reg; C(1, 2) = cc_12_reg; C(1, 3) = cc_13_reg;
    C(2, 0) = cc_20_reg; C(2, 1) = cc_21_reg; C(2, 2) = cc_22_reg; C(2, 3) = cc_23_reg;
    C(3, 0) = cc_30_reg; C(3, 1) = cc_31_reg; C(3, 2) = cc_32_reg; C(3, 3) = cc_33_reg;
}
```
|矩阵大小|gflops/sec|diff|
|---|---|---|
|40 |2.428474e+00 |0.000000e+00|
|80 |2.455883e+00 |0.000000e+00|
|120| 2.437306e+00| 0.000000e+00|
|160| 2.385064e+00| 0.000000e+00|
|200| 2.362524e+00| 0.000000e+00|
|240| 2.350163e+00| 0.000000e+00|
|280| 2.348388e+00| 0.000000e+00|
|320| 2.247154e+00| 0.000000e+00|
|360| 2.369302e+00| 0.000000e+00|
|400| 2.374904e+00| 0.000000e+00|
|440| 2.376033e+00| 0.000000e+00|
|480| 2.308397e+00| 0.000000e+00|

### MMult_4x4_7.c
优化思路同**MMult_1x4_7**，使用了指针访问，性能有一定的提升。

```c++
void AddDot4x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc)
{
    register double 
        cc_00_reg, cc_01_reg, cc_02_reg, cc_03_reg,
        cc_10_reg, cc_11_reg, cc_12_reg, cc_13_reg,
        cc_20_reg, cc_21_reg, cc_22_reg, cc_23_reg,
        cc_30_reg, cc_31_reg, cc_32_reg, cc_33_reg,

        b_p0_reg, b_p1_reg, b_p2_reg, b_p3_reg;

    cc_00_reg = 0.0; cc_01_reg = 0.0; cc_02_reg = 0.0; cc_03_reg = 0.0;
    cc_10_reg = 0.0; cc_11_reg = 0.0; cc_12_reg = 0.0; cc_13_reg = 0.0;
    cc_20_reg = 0.0; cc_21_reg = 0.0; cc_22_reg = 0.0; cc_23_reg = 0.0;
    cc_30_reg = 0.0; cc_31_reg = 0.0; cc_32_reg = 0.0; cc_33_reg = 0.0;

    double *a_0p_ptr, *a_1p_ptr, *a_2p_ptr, *a_3p_ptr;

    for (int i = 0; i < k; ++i) {
        b_p0_reg = B(i, 0);
        b_p1_reg = B(i, 1);
        b_p2_reg = B(i, 2);
        b_p3_reg = B(i, 3);

        a_0p_ptr = &A(0, i);
        a_1p_ptr = &A(1, i);
        a_2p_ptr = &A(2, i);
        a_3p_ptr = &A(3, i);

        cc_00_reg += *a_0p_ptr * b_p0_reg;
        cc_01_reg += *a_0p_ptr * b_p1_reg;
        cc_02_reg += *a_0p_ptr * b_p2_reg;
        cc_03_reg += *a_0p_ptr * b_p3_reg;
        
        cc_10_reg += *a_1p_ptr * b_p0_reg;
        cc_11_reg += *a_1p_ptr * b_p1_reg;
        cc_12_reg += *a_1p_ptr * b_p2_reg;
        cc_13_reg += *a_1p_ptr * b_p3_reg;
        
        cc_20_reg += *a_2p_ptr * b_p0_reg;
        cc_21_reg += *a_2p_ptr * b_p1_reg;
        cc_22_reg += *a_2p_ptr * b_p2_reg;
        cc_23_reg += *a_2p_ptr * b_p3_reg;
        
        cc_30_reg += *a_3p_ptr * b_p0_reg;
        cc_31_reg += *a_3p_ptr * b_p1_reg;
        cc_32_reg += *a_3p_ptr * b_p2_reg;
        cc_33_reg += *a_3p_ptr * b_p3_reg;
    }
    C(0, 0) = cc_00_reg; C(0, 1) = cc_01_reg; C(0, 2) = cc_02_reg; C(0, 3) = cc_03_reg;
    C(1, 0) = cc_10_reg; C(1, 1) = cc_11_reg; C(1, 2) = cc_12_reg; C(1, 3) = cc_13_reg;
    C(2, 0) = cc_20_reg; C(2, 1) = cc_21_reg; C(2, 2) = cc_22_reg; C(2, 3) = cc_23_reg;
    C(3, 0) = cc_30_reg; C(3, 1) = cc_31_reg; C(3, 2) = cc_32_reg; C(3, 3) = cc_33_reg;
}
```
|矩阵大小|gflops/sec|diff|
|---|---|---|
|40 |2.675921e+00 |0.000000e+00|
|80 |2.716180e+00 |0.000000e+00|
|120| 2.697806e+00| 0.000000e+00|
|160| 2.622560e+00| 0.000000e+00|
|200| 2.600798e+00| 0.000000e+00|
|240| 2.609666e+00| 0.000000e+00|
|280| 2.607494e+00| 0.000000e+00|
|320| 2.443345e+00| 0.000000e+00|
|360| 2.614944e+00| 0.000000e+00|
|400| 2.609697e+00| 0.000000e+00|
|440| 2.614179e+00| 0.000000e+00|
|480| 2.512204e+00| 0.000000e+00|

### MMult_4x4_8.c
和上个版本区别不大，只有一个优化点，在**MMult_4x4_7**中，每次访问矩阵a的值都是通过索引的形式实现，而在这个版本中通过指针累加的方式实现，并放入到寄存器当中。官方的benchmark相对于上一个版本反而下降了，可能是机器的原因，但是我在测试中取得了最好的效果。
```c++
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

        cc_00_reg += a_0p_reg * b_p0_reg;
        cc_01_reg += a_0p_reg * b_p1_reg;
        cc_02_reg += a_0p_reg * b_p2_reg;
        cc_03_reg += a_0p_reg * b_p3_reg;

        cc_10_reg += a_1p_reg * b_p0_reg;
        cc_11_reg += a_1p_reg * b_p1_reg;
        cc_12_reg += a_1p_reg * b_p2_reg;
        cc_13_reg += a_1p_reg * b_p3_reg;

        cc_20_reg += a_2p_reg * b_p0_reg;
        cc_21_reg += a_2p_reg * b_p1_reg;
        cc_22_reg += a_2p_reg * b_p2_reg;
        cc_23_reg += a_2p_reg * b_p3_reg;

        cc_30_reg += a_3p_reg * b_p0_reg;
        cc_31_reg += a_3p_reg * b_p1_reg;
        cc_32_reg += a_3p_reg * b_p2_reg;
        cc_33_reg += a_3p_reg * b_p3_reg;
    }
    C(0, 0) = cc_00_reg; C(0, 1) = cc_01_reg; C(0, 2) = cc_02_reg; C(0, 3) = cc_03_reg;
    C(1, 0) = cc_10_reg; C(1, 1) = cc_11_reg; C(1, 2) = cc_12_reg; C(1, 3) = cc_13_reg;
    C(2, 0) = cc_20_reg; C(2, 1) = cc_21_reg; C(2, 2) = cc_22_reg; C(2, 3) = cc_23_reg;
    C(3, 0) = cc_30_reg; C(3, 1) = cc_31_reg; C(3, 2) = cc_32_reg; C(3, 3) = cc_33_reg;
}
```
|矩阵大小|gflops/sec|diff|
|---|---|---|
|40 |3.176809e+00 |0.000000e+00|
|80 |3.220551e+00 |0.000000e+00|
|120| 3.177566e+00| 0.000000e+00|
|160| 3.091029e+00| 0.000000e+00|
|200| 2.988350e+00| 0.000000e+00|
|240| 3.009361e+00| 0.000000e+00|
|280| 3.016688e+00| 0.000000e+00|
|320| 2.757021e+00| 0.000000e+00|
|360| 3.046363e+00| 0.000000e+00|
|400| 3.045438e+00| 0.000000e+00|
|440| 3.038935e+00| 0.000000e+00|
|480| 2.895946e+00| 0.000000e+00|

### MMult_4x4_9.c
在循环内交换了计算顺序，差别不大。
```c++
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
```
|矩阵大小|gflops/sec|diff|
|---|---|---|
|40 |3.176809e+00 |0.000000e+00|
|80 |3.191682e+00 |0.000000e+00|
|120| 3.186355e+00| 0.000000e+00|
|160| 3.072000e+00| 0.000000e+00|
|200| 3.032720e+00| 0.000000e+00|
|240| 3.021488e+00| 0.000000e+00|
|280| 3.029708e+00| 0.000000e+00|
|320| 2.757301e+00| 0.000000e+00|
|360| 3.036635e+00| 0.000000e+00|
|400| 3.020043e+00| 0.000000e+00|
|440| 3.032008e+00| 0.000000e+00|
|480| 2.868919e+00| 0.000000e+00|

### MMuult_4x4阶段性总结
到目前为止，其优化思路和**1x4**版本中保持一致，优化结果也相对保持一致，分块+使用寄存器，能显著提升优化效果。在接下来的优化中，将会使用**instric**特性，不同平台的指令集不一样。在官方的教程中使用的是**x86_64**架构下的**SSE**系列指令集，通过使用开源的转换头文件**sse2neon.h**转换为对应的**arm**指令集，奇怪的是效果反正更差，目前还不太清楚是由什么原因造成的。

### MMult_4x4_10.c
从这个优化开始，使用了基于硬件的intrinsics指令。由于采用了向量并行计算，理论上速度会得到很大的提升，不过奇怪的是，在有些实现上性能反而有大的下降，有些实现上性能也没有得到提升。在实现中使用的并行度是2，本来想使用并行度为4的指令，不过有些指令不支持并行度为4的``double``运算。比方说``vmlaq_f32``指令没有对应的``f64``版本，而相应的``vmlaq_f64``则为两个``f64``的计算。
1. 首先使用了**乘加**指令集``vmlaq_f64``，其原型为``float64x2_t vmlaq_f64(float64x2_t a, float64x2_t b, float64x2_t c)``，实现的结果为``result = a + b * c``。计算结果如下表所示：
|矩阵大小|gflops/sec|diff|
|---|---|---|
|40 |2.329051e+00 |0.000000e+00|
|80 |2.302200e+00 |0.000000e+00|
|120| 2.223878e+00| 0.000000e+00|
|160| 2.313525e+00| 0.000000e+00|
|200| 2.240792e+00| 0.000000e+00|
|240| 2.257509e+00| 0.000000e+00|
|280| 2.255714e+00| 0.000000e+00|
|320| 2.130100e+00| 0.000000e+00|
|360| 2.266766e+00| 0.000000e+00|
|400| 2.280314e+00| 0.000000e+00|
|440| 2.272191e+00| 0.000000e+00|
|480| 2.183286e+00| 0.000000e+00|
很明显，速度相对于没有使用intrinsics，反而下降了很多。同时，跟使用``x86``平台下的``sse``指令集并通过头文件``sse2neo.h``进行翻译的效果差不多。
2. 使用``*``和``+``的操作符来代替原语，结果如下表所示：
|矩阵大小|gflops/sec|diff|
|---|---|---|
|40 |2.767568e+00 |0.000000e+00|
|80 |2.823535e+00 |0.000000e+00|
|120| 2.802258e+00| 0.000000e+00|
|160| 2.720992e+00| 0.000000e+00|
|200| 2.639411e+00| 0.000000e+00|
|240| 2.658664e+00| 0.000000e+00|
|280| 2.648988e+00| 0.000000e+00|
|320| 2.465718e+00| 0.000000e+00|
|360| 2.681553e+00| 0.000000e+00|
|400| 2.683907e+00| 0.000000e+00|
|440| 2.663642e+00| 0.000000e+00|
|480| 2.547146e+00| 0.000000e+00|
可以看到，相对于使用原语，操作符的速度明显快了很多，但还是跟上个版本有一定的差距。

### MMult_4x4_11.c
分块的思想，当矩阵越来越大的时候，会起到一定的效果。简单来说就是将A矩阵分割为一个一个小矩阵，分别和矩阵B相乘。矩阵C会不断累加阶段性的结果。可以看到当矩阵大小在``480``的时候，本次优化没有出现性能下降，但是在``MMult_4x4_11``中，有一个明显的下降。
|矩阵大小|gflops/sec|diff|
|---|---|---|
|40 |2.775068e+00 |0.000000e+00|
|80 |2.822554e+00 |0.000000e+00|
|120| 2.785320e+00| 0.000000e+00|
|160| 2.726879e+00| 0.000000e+00|
|200| 2.736622e+00| 0.000000e+00|
|240| 2.700692e+00| 0.000000e+00|
|280| 2.671392e+00| 0.000000e+00|
|320| 2.668381e+00| 0.000000e+00|
|360| 2.665486e+00| 0.000000e+00|
|400| 2.657561e+00| 0.000000e+00|
|440| 2.630739e+00| 0.000000e+00|
|480| 2.636226e+00| 0.000000e+00|

### MMult_4x4_12.c
矩阵B是按列访问的，因此是访问不连续的内存，效率比较低。在本次优化中将矩阵B的每一列提前取出来，储存在连续的内存中。理论上能加速，但是这个实现有重复的拷贝步骤，因此速度反而下降了很多。
```c++
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
```
|矩阵大小|gflops/sec|diff|
|---|---|---|
|40 |2.000000e+00 |0.000000e+00|
|80 |2.096213e+00 |0.000000e+00|
|120| 2.085225e+00| 0.000000e+00|
|160| 2.079584e+00| 0.000000e+00|
|200| 2.074902e+00| 0.000000e+00|
|240| 2.083013e+00| 0.000000e+00|
|280| 2.080791e+00| 0.000000e+00|
|320| 2.071373e+00| 0.000000e+00|
|360| 2.079932e+00| 0.000000e+00|
|400| 2.077562e+00| 0.000000e+00|
|440| 2.050563e+00| 0.000000e+00|
|480| 2.054662e+00| 0.000000e+00|


### MMult_4x4_13.c
没有了重复拷贝，相比**MMult_4x4_11**还是有相当的提升，并且在小矩阵和大矩阵上表现一致，没有出现性能的下降，反而有所提升。
|矩阵大小|gflops/sec|diff|
|---|---|---|
|40 |2.603425e+00 |0.000000e+00|
|80 |2.781347e+00 |0.000000e+00|
|120| 2.810138e+00| 0.000000e+00|
|160| 2.751070e+00| 0.000000e+00|
|200| 2.779265e+00| 0.000000e+00|
|240| 2.816614e+00| 0.000000e+00|
|280| 2.769801e+00| 0.000000e+00|
|320| 2.784826e+00| 0.000000e+00|
|360| 2.807829e+00| 0.000000e+00|
|400| 2.819500e+00| 0.000000e+00|
|440| 2.823565e+00| 0.000000e+00|
|480| 2.818712e+00| 0.000000e+00|

### MMult_4x4_14.c
矩阵A做packed, 将四行拷贝到连续的内存中，不过几乎没有提升。其原因也可以解释，矩阵A按行访问，本来就存储在行主序的内存中，加上拷贝数据存在考校，因此几乎没有优化。
|矩阵大小|gflops/sec|diff|
|---|---|---|
|40 |2.520132e+00 |0.000000e+00|
|80 |2.616416e+00 |0.000000e+00|
|120| 2.805764e+00| 0.000000e+00|
|160| 2.783751e+00| 0.000000e+00|
|200| 2.799201e+00| 0.000000e+00|
|240| 2.824587e+00| 0.000000e+00|
|280| 2.789312e+00| 0.000000e+00|
|320| 2.799373e+00| 0.000000e+00|
|360| 2.830846e+00| 0.000000e+00|
|400| 2.818755e+00| 0.000000e+00|
|440| 2.827079e+00| 0.000000e+00|
|480| 2.821542e+00| 0.000000e+00|

### MMult_4x4_15.c
减少了重复拷贝A矩阵，效果还是没什么提升
|矩阵大小|gflops/sec|diff|
|---|---|---|
|40 |2.682932e+00 |0.000000e+00|
|80 |2.776008e+00 |0.000000e+00|
|120| 2.803488e+00| 0.000000e+00|
|160| 2.791220e+00| 0.000000e+00|
|200| 2.801325e+00| 0.000000e+00|
|240| 2.832980e+00| 0.000000e+00|
|280| 2.811483e+00| 0.000000e+00|
|320| 2.780204e+00| 0.000000e+00|
|360| 2.830671e+00| 0.000000e+00|
|400| 2.819254e+00| 0.000000e+00|
|440| 2.828933e+00| 0.000000e+00|
|480| 2.814970e+00| 0.000000e+00|

### MMult_4x4 总结
比较意外的是使用了``intrinsics``之后，性能反而下降了，效果最好的是**MMult_4x4_8**，使用了寄存器和指针。可能是编译器做了对应的优化，自己操作寄存器效果并不好。同时，如果能把并行度提升到4可能会有更大的提升。
## 指令集和头文件

|header|isa|
|---|---|
|<mmintrin.h> |MMX|
|<xmmintrin.h>| SSE|
|<emmintrin.h>| SSE2|
|<pmmintrin.h>| SSE3|
|<tmmintrin.h>| SSSE3|
|<smmintrin.h>| SSE4.1|
|<nmmintrin.h>| SSE4.2|
|<ammintrin.h>| SSE4A|
|<wmmintrin.h>| AES|
|<immintrin.h>| AVX, AVX2, FMA|

## Intel / ARM intrinsics equivalence

|SSE             |ARM| Explaination|
|---|---|---|
|__m128          |float32x4_t     |4 x 32 bits floats in a vector|
|_mm_load_ps     |vld1q_f32       |load float vector from memory|
|_mm_store_ps    |vst1q_f32       |store float vector to memory|
|_mm_add_ps      |vaddq_f32       |add float vectors|