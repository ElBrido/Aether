#include "aether.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

Network* aether_create_network(int initial_capacity) {
    Network* net = (Network*)malloc(sizeof(Network));
    if (!net) return NULL;

    net->capacity = initial_capacity > 0 ? initial_capacity : 10;
    net->num_nodes = 0;
    net->nodes = (Node*)malloc(sizeof(Node) * net->capacity);

    net->dt = 0.05;
    net->decay_rate = 0.1;
    net->learning_rate = 0.005;
    net->neurogenesis_threshold = 0.8;

    return net;
}

void aether_destroy_network(Network* net) {
    if (!net) return;
    for (int i = 0; i < net->num_nodes; ++i) {
        Connection* curr = net->nodes[i].connections;
        while (curr) {
            Connection* temp = curr;
            curr = curr->next;
            free(temp);
        }
    }
    free(net->nodes);
    free(net);
}

int aether_add_node(Network* net, double frequency) {
    if (net->num_nodes >= net->capacity) {
        net->capacity *= 2;
        Node* new_nodes = (Node*)realloc(net->nodes, sizeof(Node) * net->capacity);
        if (!new_nodes) {
            fprintf(stderr, "A.E.T.H.E.R. Core Error: Memory allocation failed during neurogenesis.\n");
            return -1;
        }
        net->nodes = new_nodes;
    }

    int id = net->num_nodes;
    net->nodes[id].id = id;
    net->nodes[id].amplitude = 0.0;
    net->nodes[id].phase = 0.0;
    net->nodes[id].frequency = frequency;
    net->nodes[id].next_amplitude = 0.0;
    net->nodes[id].next_phase = 0.0;
    net->nodes[id].connections = NULL;

    net->num_nodes++;
    return id;
}

void aether_add_connection(Network* net, int from_id, int to_id, double weight, double phase_shift) {
    if (from_id < 0 || from_id >= net->num_nodes || to_id < 0 || to_id >= net->num_nodes) return;

    // Check if connection exists
    Connection* curr = net->nodes[from_id].connections;
    while(curr) {
        if(curr->target_id == to_id) {
            curr->weight = weight;
            curr->phase_shift = phase_shift;
            return;
        }
        curr = curr->next;
    }

    Connection* conn = (Connection*)malloc(sizeof(Connection));
    conn->target_id = to_id;
    conn->weight = weight;
    conn->phase_shift = phase_shift;
    conn->next = net->nodes[from_id].connections;
    net->nodes[from_id].connections = conn;
}

// Physics Engine: Forward propagation via wave interference and continuous differential equations
void aether_step(Network* net) {
    // 1. Calculate natural decay and base frequency progression
    for (int i = 0; i < net->num_nodes; ++i) {
        Node* node = &net->nodes[i];

        double d_amp = -net->decay_rate * node->amplitude;
        double d_phase = node->frequency;

        node->next_amplitude = node->amplitude + d_amp * net->dt;
        node->next_phase = node->phase + d_phase * net->dt;
    }

    // 2. Compute wave interference from connections
    for (int i = 0; i < net->num_nodes; ++i) {
        Node* source = &net->nodes[i];
        if (source->amplitude < 0.001) continue; // Optimization: Ignore inactive nodes

        Connection* conn = source->connections;
        while (conn) {
            Node* target = &net->nodes[conn->target_id];

            // Phase difference determining constructive/destructive interference
            double phase_diff = source->phase - target->phase + conn->phase_shift;
            double interference = cos(phase_diff); // 1 = fully constructive, -1 = destructive

            // Energy transfer
            double signal_amp = source->amplitude * conn->weight * interference;
            target->next_amplitude += signal_amp * net->dt;

            // Phase pulling (Kuramoto model style synchronization)
            if (target->amplitude > 0.001) {
                 double phase_pull = (source->amplitude / target->amplitude) * sin(phase_diff);
                 target->next_phase += conn->weight * phase_pull * net->dt;
            }

            conn = conn->next;
        }
    }

    // 3. Apply state updates and bound variables
    for (int i = 0; i < net->num_nodes; ++i) {
        // Activation function: Non-linear bound on amplitude (e.g., Tanh)
        net->nodes[i].amplitude = tanh(net->nodes[i].next_amplitude);
        if (net->nodes[i].amplitude < 0.0) {
            net->nodes[i].amplitude = 0.0; // Restrict to positive amplitude (like ReLU)
        }

        // Normalize phase to [-PI, PI]
        net->nodes[i].phase = net->nodes[i].next_phase;
        while (net->nodes[i].phase > M_PI) net->nodes[i].phase -= 2 * M_PI;
        while (net->nodes[i].phase < -M_PI) net->nodes[i].phase += 2 * M_PI;
    }
}

// Continuous Hebbian Plasticity
// Adjusts weights based on correlated activity and phase synchronization.
void aether_apply_hebbian_learning(Network* net) {
    for (int i = 0; i < net->num_nodes; ++i) {
        Node* source = &net->nodes[i];
        if (source->amplitude < 0.001) continue;

        Connection* conn = source->connections;
        while (conn) {
            Node* target = &net->nodes[conn->target_id];

            // Co-activation
            double co_activation = source->amplitude * target->amplitude;

            if (co_activation > 0.001) {
                // Phase correlation (1 if in phase, -1 if out of phase)
                double phase_diff = source->phase - target->phase + conn->phase_shift;
                double correlation = cos(phase_diff);

                // Hebbian update rule: w_new = w_old + learning_rate * (co_activation * correlation - decay * w_old)
                double weight_update = net->learning_rate * (co_activation * correlation - 0.01 * conn->weight);
                conn->weight += weight_update;

                // Bound weights to prevent explosion
                if (conn->weight > 5.0) conn->weight = 5.0;
                if (conn->weight < -5.0) conn->weight = -5.0;

                // Shift delay adaptation: networks learn to align delays with natural frequencies
                if (correlation > 0.5) { // Only adjust phase shift if mostly correlated
                    double phase_shift_update = net->learning_rate * sin(phase_diff); // Pull towards 0 phase diff
                    conn->phase_shift -= phase_shift_update;

                    // Normalize
                    while (conn->phase_shift > M_PI) conn->phase_shift -= 2 * M_PI;
                    while (conn->phase_shift < -M_PI) conn->phase_shift += 2 * M_PI;
                }
            }
            conn = conn->next;
        }
    }
}

// Neurogenesis: Adds a new node if the error exceeds the threshold.
// Returns the ID of the new node, or -1 if none was created.
int aether_neurogenesis_check(Network* net, double error_signal) {
    if (error_signal > net->neurogenesis_threshold) {
        // Base frequency randomized slightly to allow the network to discover new resonance modes
        double new_freq = ((double)rand() / RAND_MAX) * 2.0 * M_PI;

        int new_node_id = aether_add_node(net, new_freq);
        if (new_node_id >= 0) {
            printf("[A.E.T.H.E.R] NEUROGÉNESIS ACTIVADA. Nuevo nodo creado: %d (Error: %.4f > Umbral: %.4f)\n",
                    new_node_id, error_signal, net->neurogenesis_threshold);

            // Connect the new node sparsely to the network
            int num_connections = (net->num_nodes - 1) * 0.1; // Connect to 10% of existing nodes
            if (num_connections < 1) num_connections = 1;
            if (num_connections > 5) num_connections = 5; // Cap initial connections

            for(int i = 0; i < num_connections; i++) {
                int target = rand() % (net->num_nodes - 1);
                double init_weight = ((double)rand() / RAND_MAX) * 0.2 - 0.1; // [-0.1, 0.1]
                double init_phase = ((double)rand() / RAND_MAX) * 2.0 * M_PI - M_PI; // [-PI, PI]

                aether_add_connection(net, new_node_id, target, init_weight, init_phase);

                // Reciprocal connection
                double init_weight_recip = ((double)rand() / RAND_MAX) * 0.2 - 0.1;
                double init_phase_recip = ((double)rand() / RAND_MAX) * 2.0 * M_PI - M_PI;
                aether_add_connection(net, target, new_node_id, init_weight_recip, init_phase_recip);
            }
        }
        return new_node_id;
    }
    return -1;
}

// Background Optimization (Desfragmentación y Ensoñación Estructurada)
// Runs when the system is idle to optimize memory, prune weak connections,
// and structure knowledge for faster inference.
void aether_background_optimization(Network* net) {
    printf("[A.E.T.H.E.R] Iniciando Rutina de Optimización en Segundo Plano (Ensoñación Estructurada)...\n");
    int pruned_connections = 0;

    for (int i = 0; i < net->num_nodes; ++i) {
        Connection* curr = net->nodes[i].connections;
        Connection* prev = NULL;

        while (curr) {
            // 1. Desfragmentación: Prune extremely weak connections (close to 0 weight)
            if (fabs(curr->weight) < 0.005) {
                Connection* to_delete = curr;
                if (prev) {
                    prev->next = curr->next;
                } else {
                    net->nodes[i].connections = curr->next;
                }
                curr = curr->next;
                free(to_delete);
                pruned_connections++;
                continue;
            }

            // 2. Ensoñación Estructurada: Normalize and consolidate weights
            // Slightly decay all weights to prevent saturation and maintain plasticity
            curr->weight *= 0.99;

            prev = curr;
            curr = curr->next;
        }

        // 3. Normalize node frequencies slightly towards harmonics to stabilize the engine
        net->nodes[i].frequency = fmod(net->nodes[i].frequency, 2.0 * M_PI);
    }

    printf("[A.E.T.H.E.R] Optimización completada. %d conexiones débiles eliminadas. Red consolidada y lista.\n", pruned_connections);
}
