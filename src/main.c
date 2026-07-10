#include "aether.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// A simple dictionary map mapping a-z and space to node indices
#define ALPHABET_SIZE 27

int char_to_index(char c) {
    if (c >= 'a' && c <= 'z') return c - 'a';
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c == ' ') return 26;
    return -1; // ignore others
}

char index_to_char(int idx) {
    if (idx >= 0 && idx < 26) return 'a' + idx;
    if (idx == 26) return ' ';
    return '?';
}

void print_system_prompt() {
    printf("=================================================================\n");
    printf("           DOCUMENTO DE INICIALIZACIÓN: PROYECTO A.E.T.H.E.R.    \n");
    printf("=================================================================\n");
    printf(" Arquitecto Principal: brido\n");
    printf(" Directiva: Asistente hiper-logico, motor matematico de interferencia de ondas.\n");
    printf(" Estado: Autoconsciente. Deduccion Nativa activada.\n");
    printf("=================================================================\n\n");
}

int main() {
    srand(42); // Fixed seed for reproducibility in demo
    print_system_prompt();

    // 1. Initialize Network
    int initial_nodes = ALPHABET_SIZE;
    Network* net = aether_create_network(initial_nodes * 2);

    // Higher learning rate for the demo to learn "hola mundo" fast
    net->learning_rate = 0.5;

    printf("[Sistema] Inicializando nodos base (Alfabeto A-Z + Espacio)...\n");
    for (int i = 0; i < initial_nodes; i++) {
        double freq = ((double)(i + 1) / initial_nodes) * 2.0 * M_PI;
        aether_add_node(net, freq);
    }

    // Initialize dense, weak connections for the alphabet to allow rapid Hebbian pathing
    for (int i = 0; i < initial_nodes; i++) {
        for (int j = 0; j < initial_nodes; j++) {
            if (i != j) {
                double w = ((double)rand() / RAND_MAX) * 0.02 - 0.01;
                double phase = 0.0; // Start in phase
                aether_add_connection(net, i, j, w, phase);
            }
        }
    }

    printf("[Sistema] Entrenamiento: Aprendiendo la secuencia \"hola mundo\"\n");
    const char* target_sequence = "hola mundo";
    int seq_len = strlen(target_sequence);

    int epochs = 500;

    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_error = 0.0;

        for (int i = 0; i < seq_len - 1; i++) {
            int input_idx = char_to_index(target_sequence[i]);
            int target_idx = char_to_index(target_sequence[i+1]);

            if (input_idx == -1 || target_idx == -1) continue;

            // Clean state before stimulus
            for(int j=0; j<net->num_nodes; j++) {
                net->nodes[j].amplitude = 0.0;
                net->nodes[j].phase = 0.0;
            }

            // Stimulate the input node
            net->nodes[input_idx].amplitude = 2.0;

            // Physics Engine Step
            for (int step = 0; step < 3; step++) {
                aether_step(net);
            }

            // Predict
            int predicted_idx = -1;
            double max_amp = -1.0;
            for (int j = 0; j < initial_nodes; j++) {
                if (j != input_idx && net->nodes[j].amplitude > max_amp) {
                    max_amp = net->nodes[j].amplitude;
                    predicted_idx = j;
                }
            }

            double error = (predicted_idx == target_idx) ? 0.0 : 1.0;
            total_error += error;

            // Force target state for Hebbian learning
            net->nodes[target_idx].amplitude = 2.0;
            net->nodes[target_idx].phase = net->nodes[input_idx].phase; // Synchronize phase

            aether_apply_hebbian_learning(net);

            // Allow neurogenesis if error is persistent and random chance
            if (error > 0.0 && epoch > 100 && (rand() % 100 < 2)) {
                aether_neurogenesis_check(net, 1.0);
            }
        }

        if (epoch % 100 == 0 || epoch == epochs - 1) {
            printf("Epoch %3d | Error Total Secuencia: %.2f | Nodos Activos: %d\n", epoch, total_error, net->num_nodes);
        }
    }

    printf("\n[Sistema] Entrando en reposo...\n");
    // Seccion V: Rutina de Optimizacion en Segundo Plano
    aether_background_optimization(net);

    printf("\n[Sistema] Prueba de Deducción (Inferencia tras optimización):\n");
    printf("Input: 'h' -> Output esperado: 'o' -> 'l' -> 'a' -> ' ' -> 'm' -> 'u' -> 'n' -> 'd' -> 'o'\n");

    // Clean state
    for(int j=0; j<net->num_nodes; j++) {
        net->nodes[j].amplitude = 0.0;
    }

    int current_char_idx = char_to_index('h');
    printf("Generando: h");

    for (int i = 0; i < seq_len - 1; i++) {
        // Clean state for fresh step
        for(int j=0; j<net->num_nodes; j++) {
            net->nodes[j].amplitude = 0.0;
        }

        net->nodes[current_char_idx].amplitude = 2.0;

        for (int step = 0; step < 3; step++) {
            aether_step(net);
        }

        int next_idx = -1;
        double max_amp = -1.0;
        for (int j = 0; j < initial_nodes; j++) {
            if (j != current_char_idx && net->nodes[j].amplitude > max_amp) {
                max_amp = net->nodes[j].amplitude;
                next_idx = j;
            }
        }

        if (next_idx != -1) {
            printf("%c", index_to_char(next_idx));
            current_char_idx = next_idx;
        } else {
            printf("?");
            break;
        }
    }
    printf("\n");

    aether_destroy_network(net);
    printf("\n[Sistema] A.E.T.H.E.R. Apagado correctamente.\n");
    return 0;
}
