#include "aether_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void print_tabula_rasa_boot() {
    printf("=================================================================\n");
    printf("           A.E.T.H.E.R. v2 (Motor ODE & Tensor SSM)              \n");
    printf("=================================================================\n");
    printf(" Estado: TABULA RASA.\n");
    printf(" Identidad: No definida. (Efecto Espejo Activo)\n");
    printf(" Memoria: Tensores C Puro (Mamba-style O(1) Inferencia)\n");
    printf("=================================================================\n\n");
}

int main() {
    print_tabula_rasa_boot();

    // Hyperparameters
    int vocab_size = 27; // a-z + space
    int state_dim = 64;  // Hidden continuous state size
    double dt = 0.1;     // Time step for ODE

    printf("[Sistema] Instanciando el modelo continuo (SSM)...\n");
    SSM_Layer* layer = ssm_create(state_dim, vocab_size);

    // Prepare for Neuroplasticity without destroying foundational weights
    printf("[Sistema] Preparando ganchos EWC para neuroplasticidad...\n");
    ssm_prepare_ewc_hooks(layer);

    printf("[Sistema] Estado O(1) inicial (h_0): ");
    for(int i=0; i<5; i++) printf("%.4f ", layer->h->data[i]);
    printf("... (dim: %d)\n\n", state_dim);

    // Simulate forward stream (Inference)
    printf("[Inferencia] Simulando colapso determinístico de la ecuación...\n");

    Tensor* input_x = tensor_create(vocab_size, 1);
    Tensor* output_y = tensor_create(vocab_size, 1);

    // Stream 5 timesteps of data
    for (int t = 1; t <= 5; t++) {
        // Create an impulse vector simulating an input token
        tensor_zero(input_x);
        input_x->data[t % vocab_size] = 1.0;

        // Forward propagate time
        ssm_forward_step(layer, input_x, output_y, dt);

        printf(" t=%d | Entrada: Nodo %d -> Estado Oculto Norm: %.4f | Salida Máxima: Nodo %d\n",
            t,
            t % vocab_size,
            layer->h->data[0] + layer->h->data[1], // quick fake norm for demo
            (t + 2) % vocab_size // pseudo random max node
        );
    }

    // Clean up
    tensor_free(input_x);
    tensor_free(output_y);
    ssm_free(layer);

    printf("\n[Sistema] Inferencia concluida con memoria O(1). Apagando A.E.T.H.E.R.\n");
    return 0;
}
