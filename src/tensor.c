#include "aether_core.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

Tensor* tensor_create(int rows, int cols) {
    if (rows <= 0 || cols <= 0) return NULL;
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->rows = rows;
    t->cols = cols;
    t->data = (double*)calloc(rows * cols, sizeof(double));
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

void tensor_randomize(Tensor* t, double min_val, double max_val) {
    if (!t || !t->data) return;
    int size = t->rows * t->cols;
    double range = max_val - min_val;
    for (int i = 0; i < size; i++) {
        t->data[i] = min_val + ((double)rand() / RAND_MAX) * range;
    }
}

void tensor_zero(Tensor* t) {
    if (!t || !t->data) return;
    int size = t->rows * t->cols;
    for (int i = 0; i < size; i++) {
        t->data[i] = 0.0;
    }
}

void tensor_add(Tensor* out, Tensor* a, Tensor* b) {
    if (!out || !a || !b) return;
    if (out->rows != a->rows || out->rows != b->rows ||
        out->cols != a->cols || out->cols != b->cols) {
        fprintf(stderr, "Tensor Add Error: Dimension mismatch\n");
        return;
    }
    int size = out->rows * out->cols;
    for (int i = 0; i < size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
}

void tensor_matmul(Tensor* out, Tensor* a, Tensor* b) {
    if (!out || !a || !b) return;
    if (a->cols != b->rows || out->rows != a->rows || out->cols != b->cols) {
        fprintf(stderr, "Tensor MatMul Error: Dimension mismatch (%dx%d) * (%dx%d) -> (%dx%d)\n",
                a->rows, a->cols, b->rows, b->cols, out->rows, out->cols);
        return;
    }

    // Brutal efficiency approach for basic pure C (O(N^3))
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < a->cols; k++) {
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
            out->data[i * out->cols + j] = sum;
        }
    }
}

void tensor_scale(Tensor* t, double scalar) {
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
