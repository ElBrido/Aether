#include "aether_core.h"
#include <stdlib.h>

AetherEngine* aether_create(int vocab_size, int hidden_dim) {
    AetherEngine* engine = (AetherEngine*)malloc(sizeof(AetherEngine));
    engine->vocab_size = vocab_size;
    engine->hidden_dim = hidden_dim;

    // Create sub-components
    engine->ode = ode_func_create(hidden_dim, vocab_size);
    engine->ssm = ssm_create(hidden_dim, hidden_dim); // SSM processes ODE's output

    engine->W_out = tensor_create(vocab_size, hidden_dim);
    engine->b_out = tensor_create(vocab_size, 1);

    engine->logits = tensor_create(vocab_size, 1);

    tensor_randomize(engine->W_out, -0.1f, 0.1f);
    tensor_zero(engine->b_out);

    return engine;
}

void aether_free(AetherEngine* engine) {
    if (!engine) return;
    ode_func_free(engine->ode);
    ssm_free(engine->ssm);
    tensor_free(engine->W_out);
    tensor_free(engine->b_out);
    tensor_free(engine->logits);
    free(engine);
}

// Full forward pass: ODE -> SSM -> Linear Projection
void aether_forward(AetherEngine* engine, const Tensor* x_t) {
    // 1. ODE Step (Simulate semantic dynamics globally)
    // h evolves continuously using RK4.
    // We use a dummy h for ODE that will be fed into SSM.
    Tensor* ode_h = tensor_create(engine->hidden_dim, 1);
    tensor_zero(ode_h); // In reality this could be recurrent, here it represents transient token dynamics

    float dt = 0.1f; // Integration step
    // Run 10 steps to cover t in [0.0, 1.0]
    for (int step = 0; step < 10; step++) {
        rk4_step(engine->ode, ode_h, x_t, dt);
    }

    // 2. SSM Layer (Memory filtering and context extraction)
    Tensor* ssm_out = tensor_create(engine->hidden_dim, 1);
    ssm_forward_step(engine->ssm, ode_h, ssm_out);

    // 3. Linear Projection (logits)
    tensor_matmul(engine->logits, engine->W_out, ssm_out);
    tensor_add(engine->logits, engine->logits, engine->b_out);

    tensor_free(ode_h);
    tensor_free(ssm_out);
}
