/*=============================
 Arm neon instrinsics learning
 ==============================*/

#include <stdio.h>
#include <arm_neon.h>

void test_vaddq()
{
    float32x4_t a = {1.0, 2.0, 3.0, 4.0};
    float32x4_t b = {2.0, 3.0, 4.0, 5.0};
    float32x4_t c = {0.0, 0.0, 0.0, 0.0};
    c = vaddq_f32(a, b);
    float result[4];
    vst1q_f32(&result[0], c); // 把数据从寄存器取回到内存数组中
    printf("vaddq: \n");
    for (int i = 0; i < 4; ++i) {
        printf("%f ", result[i]);
    }
    printf("\n");
}

void test_vmulq()
{
    float32x4_t a = {1.0, 2.0, 3.0, 4.0};
    float32x4_t b = {2.0, 3.0, 4.0, 5.0};
    float32x4_t c = {0.0, 0.0, 0.0, 0.0};
    c = vmulq_f32(a, b);
    float result[4];
    vst1q_f32(&result[0], c);
    printf("vmulq: \n");
    for (int i = 0; i < 4; ++i) {
        printf("%f ", result[i]);
    }
    printf("\n");
}

void test_vsubq()
{
    float32x4_t a = {1.0, 2.0, 3.0, 4.0};
    float32x4_t b = {2.0, 3.0, 4.0, 5.0};
    float32x4_t c = {0.0, 0.0, 0.0, 0.0};
    c = vsubq_f32(a, b);
    float result[4];
    vst1q_f32(&result[0], c);
    printf("vsubq: \n");
    for (int i = 0; i < 4; ++i) {
        printf("%f ", result[i]);
    }
    printf("\n");
}

void test_vmlaq()
{
    float32x4_t a = {1.0, 2.0, 3.0, 4.0};
    float32x4_t b = {2.0, 3.0, 4.0, 5.0};
    float32x4_t c = {2.0, 2.0, 2.0, 2.0};
    c = vmlaq_f32(c, a, b);
    float result[4];
    vst1q_f32(&result[0], c);
    printf("vmlaq: \n");
    for (int i = 0; i < 4; ++i) {
        printf("%f ", result[i]);
    }
    printf("\n");
}

void test_vmulq_n()
{
    float32x4_t a = {1.0, 2.0, 3.0, 4.0};
    float32_t b = 2.0;
    float32x4_t c = {0.0, 0.0, 0.0, 0.0};
    c = vmulq_n_f32(a, b);
    float result[4];
    vst1q_f32(&result[0], c);
    printf("vmulq_n: \n");
    for (int i = 0; i < 4; ++i) {
        printf("%f ", result[i]);
    }
    printf("\n");
}

void test_vmlaq_n()
{
    float32x4_t a = {1.0, 2.0, 3.0, 4.0};
    float32_t b = 2.0;
    float32x4_t c = {0.0, 0.0, 0.0, 0.0};
    c = vmulq_n_f32(a, b);
    float result[4];
    vst1q_f32(&result[0], c);
    printf("vmulq_n: \n");
    for (int i = 0; i < 4; ++i) {
        printf("%f ", result[i]);
    }
    printf("\n");
}

int main()
{
    test_vaddq();
    test_vmulq();
    test_vsubq();
    test_vmlaq();
    test_vmulq_n();
}
