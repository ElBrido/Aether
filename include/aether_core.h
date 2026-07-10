#ifndef AETHER_CORE_H
#define AETHER_CORE_H

#include <stddef.h>
#include <stdbool.h>

// ---------------------------------------------------------
// TENSOR SYSTEM (Pure C, No Dependencies)
// ---------------------------------------------------------
typedef struct Tensor {
    int rows;
    int cols;
    double* data; // Contiguous 1D array for 2D data
} Tensor;

// Memory Management
Tensor* tensor_create(int rows, int cols);
void tensor_free(Tensor* t);
void tensor_randomize(Tensor* t, double min_val, double max_val);
void tensor_zero(Tensor* t);

// Math Operations
void tensor_add(Tensor* out, Tensor* a, Tensor* b);
void tensor_matmul(Tensor* out, Tensor* a, Tensor* b); // out = A * B
void tensor_scale(Tensor* t, double scalar);
void tensor_copy(Tensor* dst, Tensor* src);

// ---------------------------------------------------------
// STATE SPACE MODEL (SSM) & NEURAL ODE CORE
// ---------------------------------------------------------
// Represents a single continuous-time layer h_t = A * h_{t-1} + B * x_t
typedef struct SSM_Layer {
    int state_dim;
    int input_dim;

    // Core Matrices
    Tensor* A; // Transition matrix (state_dim x state_dim)
    Tensor* B; // Input matrix (state_dim x input_dim)
    Tensor* C; // Output projection (input_dim x state_dim)

    // Continuous State (O(1) memory during inference)
    Tensor* h; // Hidden state (state_dim x 1)

    // Efficient inference buffers (allocated once to avoid malloc in hot loop)
    Tensor* buf_Ah;
    Tensor* buf_Bx;
    Tensor* buf_dh;

    // Neuroplasticity Hooks (EWC / Hebbian)
    Tensor* A_fisher; // Fisher information for A to prevent catastrophic forgetting
    Tensor* A_anchor; // Base weights of A for EWC penalty
} SSM_Layer;

SSM_Layer* ssm_create(int state_dim, int input_dim);
void ssm_free(SSM_Layer* layer);

// Forward propagation in continuous time (ODE Step)
// dt: time step for Euler integration
void ssm_forward_step(SSM_Layer* layer, Tensor* x_t, Tensor* out_t, double dt);

// Initializes EWC anchors to current weights
void ssm_prepare_ewc_hooks(SSM_Layer* layer);

#endif // AETHER_CORE_H
