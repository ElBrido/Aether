#include "aether_core.h"
#include <stdlib.h>
#include <stdio.h>

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
    int size = t->rows * t->cols;
    for (int i = 0; i < size; i++) {
        t->data[i] = 0.0f;
    }
}

void tensor_add(Tensor* out, Tensor* a, Tensor* b) {
    if (!out || !a || !b) return;
    int size = out->rows * out->cols;
    for (int i = 0; i < size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
}

void tensor_matmul(Tensor* out, Tensor* a, Tensor* b) {
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

void tensor_copy(Tensor* dst, Tensor* src) {
    if (!dst || !src || dst->rows != src->rows || dst->cols != src->cols) return;
    int size = dst->rows * dst->cols;
    for (int i = 0; i < size; i++) {
        dst->data[i] = src->data[i];
    }
}
