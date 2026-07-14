#include "aether_core.h"
#include <stdlib.h>
#include <math.h>

SSMLayer* ssm_create(int state_dim, int input_dim) {
    SSMLayer* layer = (SSMLayer*)malloc(sizeof(SSMLayer));
    layer->state_dim = state_dim;
    layer->input_dim = input_dim;

    layer->A = tensor_create(state_dim, state_dim);
    layer->B = tensor_create(state_dim, input_dim);
    layer->C = tensor_create(input_dim, state_dim);
    layer->D = tensor_create(input_dim, input_dim);

    layer->delta = tensor_create(1, 1);
    layer->delta->data[0] = 0.1f; // base time scale

    layer->A_bar = tensor_create(state_dim, state_dim);
    layer->B_bar = tensor_create(state_dim, input_dim);

    layer->h = tensor_create(state_dim, 1);
    tensor_zero(layer->h);

    layer->buf_Ah = tensor_create(state_dim, 1);
    layer->buf_Bx = tensor_create(state_dim, 1);

    // HiPPO Initialization for A (negative values for stability)
    tensor_zero(layer->A);
    for (int i = 0; i < state_dim; i++) {
        layer->A->data[i * state_dim + i] = -(float)(i + 1);
    }

    tensor_randomize(layer->B, -0.1f, 0.1f);
    tensor_randomize(layer->C, -0.1f, 0.1f);
    tensor_randomize(layer->D, -0.05f, 0.05f);

    ssm_discretize(layer); // Initial discretization

    return layer;
}

void ssm_free(SSMLayer* layer) {
    if (!layer) return;
    tensor_free(layer->A);
    tensor_free(layer->B);
    tensor_free(layer->C);
    tensor_free(layer->D);
    tensor_free(layer->delta);
    tensor_free(layer->A_bar);
    tensor_free(layer->B_bar);
    tensor_free(layer->h);
    tensor_free(layer->buf_Ah);
    tensor_free(layer->buf_Bx);
    free(layer);
}

// Computes ā = exp(ΔA) and b̄ = (ā-1)/A * B (Zero-Order Hold)
void ssm_discretize(SSMLayer* layer) {
    float dt = layer->delta->data[0];

    // Simplification for diagonal A matrix
    for (int i = 0; i < layer->state_dim; i++) {
        float a_val = layer->A->data[i * layer->state_dim + i];

        // A_bar = exp(delta * A)
        float a_bar = expf(dt * a_val);
        layer->A_bar->data[i * layer->state_dim + i] = a_bar;

        // B_bar = (exp(delta * A) - 1) / A * B
        float scalar = (a_bar - 1.0f) / a_val;
        for (int j = 0; j < layer->input_dim; j++) {
            layer->B_bar->data[i * layer->input_dim + j] = layer->B->data[i * layer->input_dim + j] * scalar;
        }
    }
}

// h_t = ā ⊙ h_{t-1} + b̄ ⊙ Bx
// y_t = C · h_t + D · x
void ssm_forward_step(SSMLayer* layer, const Tensor* x_t, Tensor* out_t) {
    // 1. h_t = A_bar * h_{t-1} + B_bar * x_t
    tensor_matmul(layer->buf_Ah, layer->A_bar, layer->h);
    tensor_matmul(layer->buf_Bx, layer->B_bar, (Tensor*)x_t);
    tensor_add(layer->h, layer->buf_Ah, layer->buf_Bx);

    // 2. y_t = C * h_t + D * x_t
    Tensor* y_temp = tensor_create(layer->input_dim, 1);
    Tensor* Dx = tensor_create(layer->input_dim, 1);

    tensor_matmul(y_temp, layer->C, layer->h);
    tensor_matmul(Dx, layer->D, (Tensor*)x_t);

    tensor_add(out_t, y_temp, Dx);

    tensor_free(y_temp);
    tensor_free(Dx);
}
