#include "aether_core.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

Tensor* tensor_create(int rows, int cols) {
    if (rows <= 0 || cols <= 0) return NULL;
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->rows = rows;
    t->cols = cols;
    t->data = (float*)calloc(rows * cols, sizeof(float));
    if (!t->data) {
        free(t);
        return NULL;
    }
    return t;
}

void tensor_free(Tensor* t) {
    if (!t) return;
    if (t->data) free(t->data);
    free(t);
}

void tensor_randomize(Tensor* t, float min_val, float max_val) {
    if (!t || !t->data) return;
    int size = t->rows * t->cols;
    float range = max_val - min_val;
    for (int i = 0; i < size; i++) {
        t->data[i] = min_val + ((float)rand() / RAND_MAX) * range;
    }
}

void tensor_zero(Tensor* t) {
    if (!t || !t->data) return;
    memset(t->data, 0, t->rows * t->cols * sizeof(float));
}

void tensor_add(Tensor* out, const Tensor* a, const Tensor* b) {
    if (!out || !a || !b) return;
    int size = out->rows * out->cols;
    for (int i = 0; i < size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
}

void tensor_matmul(Tensor* out, const Tensor* a, const Tensor* b) {
    if (!out || !a || !b) return;
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a->cols; k++) {
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
            out->data[i * out->cols + j] = sum;
        }
    }
}

void tensor_scale(Tensor* t, float scalar) {
    if (!t || !t->data) return;
    int size = t->rows * t->cols;
    for (int i = 0; i < size; i++) {
        t->data[i] *= scalar;
    }
}

void tensor_copy(Tensor* dst, const Tensor* src) {
    if (!dst || !src || dst->rows != src->rows || dst->cols != src->cols) return;
    memcpy(dst->data, src->data, dst->rows * dst->cols * sizeof(float));
}

void tensor_rmsnorm(Tensor* out, const Tensor* in, const Tensor* weight, float eps) {
    if (!out || !in) return;
    int n = in->rows * in->cols;

    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        float x = in->data[i];
        sum_sq += x * x;
    }
    float inv_rms = 1.0f / sqrtf((sum_sq / n) + eps);

    for (int i = 0; i < n; i++) {
        float w = (weight && weight->data) ? weight->data[i] : 1.0f;
        out->data[i] = in->data[i] * inv_rms * w;
    }
}

void tensor_silu(Tensor* out, const Tensor* in) {
    if (!out || !in) return;
    int size = in->rows * in->cols;
    for (int i = 0; i < size; i++) {
        float x = in->data[i];
        float sig = 1.0f / (1.0f + expf(-x));
        out->data[i] = x * sig;
    }
}

void tensor_sigmoid(Tensor* out, const Tensor* in) {
    if (!out || !in) return;
    int size = in->rows * in->cols;
    for (int i = 0; i < size; i++) {
        out->data[i] = 1.0f / (1.0f + expf(-in->data[i]));
    }
}

void tensor_tanh(Tensor* out, const Tensor* in) {
    if (!out || !in) return;
    int size = in->rows * in->cols;
    for (int i = 0; i < size; i++) {
        out->data[i] = tanhf(in->data[i]);
    }
}

void tensor_softplus(Tensor* out, const Tensor* in) {
    if (!out || !in) return;
    int size = in->rows * in->cols;
    for (int i = 0; i < size; i++) {
        float x = in->data[i];
        if (x > 20.0f) {
            out->data[i] = x;
        } else if (x < -20.0f) {
            out->data[i] = expf(x);
        } else {
            out->data[i] = log1pf(expf(x));
        }
    }
}

void tensor_print(const Tensor* t, const char* name, int max_elems) {
    if (!t || !t->data) {
        printf("%s: (null)\n", name);
        return;
    }
    int size = t->rows * t->cols;
    int n = (max_elems > 0 && max_elems < size) ? max_elems : size;
    printf("%s [%d x %d]: ", name, t->rows, t->cols);
    for (int i = 0; i < n; i++) {
        printf("%.6f ", t->data[i]);
    }
    if (n < size) printf("...");
    printf("\n");
}
