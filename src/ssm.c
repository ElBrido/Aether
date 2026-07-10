#include "aether_core.h"
#include <stdlib.h>
#include <math.h>

SSM_Layer* ssm_create(int state_dim, int input_dim) {
    SSM_Layer* layer = (SSM_Layer*)malloc(sizeof(SSM_Layer));
    if (!layer) return NULL;

    layer->state_dim = state_dim;
    layer->input_dim = input_dim;

    // Core parameters
    layer->A = tensor_create(state_dim, state_dim);
    layer->B = tensor_create(state_dim, input_dim);
    layer->C = tensor_create(input_dim, state_dim); // Project back to input size

    // Continuous hidden state h (O(1) Memory during inference, it just overwrites)
    layer->h = tensor_create(state_dim, 1);

    // Efficient inference buffers
    layer->buf_Ah = tensor_create(state_dim, 1);
    layer->buf_Bx = tensor_create(state_dim, 1);
    layer->buf_dh = tensor_create(state_dim, 1);

    // EWC Hooks initialize as NULL until needed
    layer->A_fisher = NULL;
    layer->A_anchor = NULL;

    // Initialization: Small random weights, A matrix negative for stable ODE decay
    tensor_randomize(layer->A, -0.1, 0.0);
    tensor_randomize(layer->B, -0.1, 0.1);
    tensor_randomize(layer->C, -0.1, 0.1);
    tensor_zero(layer->h); // Start at rest

    return layer;
}

void ssm_free(SSM_Layer* layer) {
    if (!layer) return;
    tensor_free(layer->A);
    tensor_free(layer->B);
    tensor_free(layer->C);
    tensor_free(layer->h);
    tensor_free(layer->buf_Ah);
    tensor_free(layer->buf_Bx);
    tensor_free(layer->buf_dh);
    if (layer->A_fisher) tensor_free(layer->A_fisher);
    if (layer->A_anchor) tensor_free(layer->A_anchor);
    free(layer);
}

// Forward pass for Neural ODE / Continuous SSM: dh/dt = Ah + Bx -> y = Ch
void ssm_forward_step(SSM_Layer* layer, Tensor* x_t, Tensor* out_t, double dt) {
    // 1. Calculate continuous derivative dh = (A * h + B * x) * dt

    tensor_matmul(layer->buf_Ah, layer->A, layer->h);
    tensor_matmul(layer->buf_Bx, layer->B, x_t);

    tensor_add(layer->buf_dh, layer->buf_Ah, layer->buf_Bx);
    tensor_scale(layer->buf_dh, dt);

    // 2. Euler Integration Step: h_{t} = h_{t-1} + dh
    tensor_add(layer->h, layer->h, layer->buf_dh);

    // Apply non-linearity (e.g. SiLU/Swish to hidden state) for capability
    for(int i=0; i<layer->state_dim; i++) {
        double val = layer->h->data[i];
        layer->h->data[i] = val / (1.0 + exp(-val)); // Sigmoid * val
    }

    // 3. Project to output: y = C * h
    tensor_matmul(out_t, layer->C, layer->h);
}

// Prepares the model for Plasticity / Continuous learning by anchoring current weights
void ssm_prepare_ewc_hooks(SSM_Layer* layer) {
    if (!layer->A_anchor) {
        layer->A_anchor = tensor_create(layer->state_dim, layer->state_dim);
        layer->A_fisher = tensor_create(layer->state_dim, layer->state_dim);
    }

    // Save current A state as the anchor
    tensor_copy(layer->A_anchor, layer->A);

    // In a real EWC, Fisher info is computed via gradients.
    // Here we initialize it as 1.0 (flat importance) for the hook demonstration.
    tensor_zero(layer->A_fisher);
    tensor_add(layer->A_fisher, layer->A_fisher, layer->A_fisher); // Zero out
    for (int i=0; i < layer->state_dim * layer->state_dim; i++) {
        layer->A_fisher->data[i] = 1.0;
    }
}
