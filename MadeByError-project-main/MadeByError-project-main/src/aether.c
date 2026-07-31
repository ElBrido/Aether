#include "aether_core.h"
#include <stdlib.h>
#include <string.h>

AetherEngine* aether_create(int vocab_size, int hidden_dim, int ssm_state_dim, int num_layers) {
    if (num_layers <= 0) num_layers = 1;

    AetherEngine* engine = (AetherEngine*)malloc(sizeof(AetherEngine));
    engine->vocab_size = vocab_size;
    engine->hidden_dim = hidden_dim;
    engine->ssm_state_dim = ssm_state_dim;
    engine->num_layers = num_layers;
    engine->chaos_sigma = 0.02f;

    engine->embedding = embedding_create(vocab_size, hidden_dim);
    engine->pos_norm_weight = tensor_create(hidden_dim, 1);

    engine->crof_layers = (CROFLayer**)malloc(num_layers * sizeof(CROFLayer*));
    for (int l = 0; l < num_layers; l++) {
        engine->crof_layers[l] = crof_layer_create(hidden_dim, ssm_state_dim, 1.0f);
    }

    engine->final_norm_weight = tensor_create(hidden_dim, 1);

    engine->W_out = tensor_create(vocab_size, hidden_dim);
    engine->b_out = tensor_create(vocab_size, 1);

    engine->logits    = tensor_create(vocab_size, 1);
    engine->emb_vec   = tensor_create(hidden_dim, 1);
    engine->norm_vec  = tensor_create(hidden_dim, 1);
    engine->layer_in  = tensor_create(hidden_dim, 1);
    engine->layer_out = tensor_create(hidden_dim, 1);

    tensor_randomize(engine->W_out, -0.1f, 0.1f);
    tensor_zero(engine->b_out);

    return engine;
}

void aether_free(AetherEngine* engine) {
    if (!engine) return;
    embedding_free(engine->embedding);
    tensor_free(engine->pos_norm_weight);
    if (engine->crof_layers) {
        for (int l = 0; l < engine->num_layers; l++) {
            crof_layer_free(engine->crof_layers[l]);
        }
        free(engine->crof_layers);
    }
    tensor_free(engine->final_norm_weight);
    tensor_free(engine->W_out);
    tensor_free(engine->b_out);
    tensor_free(engine->logits);
    tensor_free(engine->emb_vec);
    tensor_free(engine->norm_vec);
    tensor_free(engine->layer_in);
    tensor_free(engine->layer_out);
    free(engine);
}

void aether_reset(AetherEngine* engine) {
    if (!engine) return;
    for (int l = 0; l < engine->num_layers; l++) {
        crof_layer_reset(engine->crof_layers[l]);
    }
}

void aether_forward_token(AetherEngine* engine, int token_id) {
    // 1. Embedding lookup
    embedding_lookup(engine->embedding, token_id, engine->emb_vec);

    // 2. Positional RMSNorm
    tensor_rmsnorm(engine->layer_in, engine->emb_vec, engine->pos_norm_weight, 1e-6f);

    // 3. Multi-layer CROF stack
    for (int l = 0; l < engine->num_layers; l++) {
        crof_layer_step(engine->crof_layers[l], engine->layer_in, engine->layer_out);
        tensor_copy(engine->layer_in, engine->layer_out);
    }

    // 4. Final RMSNorm
    tensor_rmsnorm(engine->norm_vec, engine->layer_out, engine->final_norm_weight, 1e-6f);

    // 5. Linear Projection to Vocab Logits
    tensor_matmul(engine->logits, engine->W_out, engine->norm_vec);
    tensor_add(engine->logits, engine->logits, engine->b_out);
}
