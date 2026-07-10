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
    float* data; // Contiguous 1D array for 2D data (using float for performance)
} Tensor;

Tensor* tensor_create(int rows, int cols);
void tensor_free(Tensor* t);
void tensor_randomize(Tensor* t, float min_val, float max_val);
void tensor_zero(Tensor* t);
void tensor_add(Tensor* out, Tensor* a, Tensor* b);
void tensor_matmul(Tensor* out, Tensor* a, Tensor* b);
void tensor_scale(Tensor* t, float scalar);
void tensor_copy(Tensor* dst, Tensor* src);

// ---------------------------------------------------------
// NEURAL ODE (dh/dt = tanh(W_hh·h + W_xh·x + bias))
// ---------------------------------------------------------
typedef struct ODEFunc {
    int hidden_dim;
    int input_dim;
    Tensor* W_hh;
    Tensor* W_xh;
    Tensor* bias;

    // Buffers for RK4 to avoid mallocs
    Tensor* buf_Whh_h;
    Tensor* buf_Wxh_x;
    Tensor* buf_dhdt;
} ODEFunc;

ODEFunc* ode_func_create(int hidden_dim, int input_dim);
void ode_func_free(ODEFunc* f);
void ode_func_forward(const ODEFunc *f, const Tensor *h, float t, const Tensor *x, Tensor *dhdt);

// Integrator RK4
void rk4_step(const ODEFunc *f, Tensor *h, const Tensor *x, float dt);

// ---------------------------------------------------------
// SSM LAYER (Mamba-style Zero-Order Hold)
// ---------------------------------------------------------
typedef struct SSMLayer {
    int state_dim;
    int input_dim;

    Tensor* A; // Transition matrix
    Tensor* B; // Input projection
    Tensor* C; // Output projection
    Tensor* D; // Skip connection

    // Learnable Delta for discretization
    Tensor* delta;

    // Discretized parameters (ā = exp(ΔA), b̄ = (ā-1)/A * B)
    Tensor* A_bar;
    Tensor* B_bar;

    // Hidden continuous state O(1)
    Tensor* h;

    // Buffers
    Tensor* buf_Ah;
    Tensor* buf_Bx;
} SSMLayer;

SSMLayer* ssm_create(int state_dim, int input_dim);
void ssm_free(SSMLayer* layer);
void ssm_discretize(SSMLayer* layer);
void ssm_forward_step(SSMLayer* layer, const Tensor* x_t, Tensor* out_t);

// ---------------------------------------------------------
// ELASTIC WEIGHT CONSOLIDATION (EWC)
// ---------------------------------------------------------
typedef struct EWC_Params {
    Tensor* params;
    Tensor* fisher;
    Tensor* anchor;
} EWC_Params;

EWC_Params* ewc_create(Tensor* target_params);
void ewc_free(EWC_Params* ewc);
void ewc_snapshot(EWC_Params* ewc);
float ewc_penalty(EWC_Params* ewc, float lambda);

// ---------------------------------------------------------
// A.E.T.H.E.R. ENGINE (Composition)
// ---------------------------------------------------------
typedef struct AetherEngine {
    int vocab_size;
    int hidden_dim;

    ODEFunc* ode;
    SSMLayer* ssm;

    // Final linear projection to logits
    Tensor* W_out;
    Tensor* b_out;

    // Memory
    Tensor* logits;
} AetherEngine;

AetherEngine* aether_create(int vocab_size, int hidden_dim);
void aether_free(AetherEngine* engine);
void aether_forward(AetherEngine* engine, const Tensor* x_t);

#endif // AETHER_CORE_H
