#include "aether_core.h"
#include <stdio.h>
#include <stdlib.h>

void print_banner() {
    printf("=================================================================\n");
    printf("           A.E.T.H.E.R. v0.1 (Motor ODE + Tensor SSM)            \n");
    printf("=================================================================\n");
    printf(" Arquitectura: Tiempo Continuo (RK4) + ZOH SSM                   \n");
    printf(" Inferencia: O(1) Memoria, Determinístico, Tabula Rasa           \n");
    printf("=================================================================\n\n");
}

int main() {
    print_banner();

    int vocab_size = 32;
    int hidden_dim = 128;

    printf("[Sistema] Inicializando motor (Vocab: %d, Dim: %d)...\n", vocab_size, hidden_dim);
    AetherEngine* engine = aether_create(vocab_size, hidden_dim);

    // Simulate streaming 5 tokens
    Tensor* x = tensor_create(vocab_size, 1);

    printf("\n[Inferencia] Ejecutando pase forward continuo (ODE -> SSM -> Proyección)\n");
    for (int t = 1; t <= 5; t++) {
        // One-hot encode a dummy token
        tensor_zero(x);
        x->data[t % vocab_size] = 1.0f;

        // Execute engine forward pass
        aether_forward(engine, x);

        // Find max logit to simulate prediction
        int pred_idx = 0;
        float max_val = engine->logits->data[0];
        for (int i = 1; i < vocab_size; i++) {
            if (engine->logits->data[i] > max_val) {
                max_val = engine->logits->data[i];
                pred_idx = i;
            }
        }

        printf(" Token [%d] -> In (Idx: %2d) | Out Logit Máx: %.4f (Idx: %2d)\n",
            t, t % vocab_size, max_val, pred_idx);
    }

    // Clean up
    tensor_free(x);
    aether_free(engine);

    printf("\n[Sistema] Inferencia O(1) finalizada sin pérdidas de memoria.\n");
    return 0;
}
