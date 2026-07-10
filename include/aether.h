#ifndef AETHER_H
#define AETHER_H

#include <stddef.h>
#include <stdbool.h>

// Represents a synaptic connection between oscillators
typedef struct Connection {
    int target_id;
    double weight;
    double phase_shift; // Phase delay for wave interference
    struct Connection* next;
} Connection;

// Represents a node (oscillator) in the AETHER engine
typedef struct Node {
    int id;
    double amplitude;   // Current wave amplitude (activation)
    double phase;       // Current wave phase
    double frequency;   // Natural frequency of the oscillator

    double next_amplitude; // Buffer for simultaneous differential equation update
    double next_phase;

    Connection* connections;
} Node;

// The AETHER main network engine
typedef struct Network {
    Node* nodes;
    int num_nodes;
    int capacity;

    double dt;             // Time step for continuous equations
    double decay_rate;     // Natural decay of amplitude
    double learning_rate;  // Continuous Hebbian plasticity rate
    double neurogenesis_threshold; // Threshold to trigger dynamic network growth
} Network;

// Core Engine Initialization
Network* aether_create_network(int initial_capacity);
void aether_destroy_network(Network* net);

// Structural Methods
int aether_add_node(Network* net, double frequency);
void aether_add_connection(Network* net, int from_id, int to_id, double weight, double phase_shift);

// Physics Engine: Forward propagation via continuous differential equations
void aether_step(Network* net);

// Learning and Adaptive Growth
void aether_apply_hebbian_learning(Network* net);
int aether_neurogenesis_check(Network* net, double error_signal);

// Background Optimization (Desfragmentación y Ensoñación Estructurada)
void aether_background_optimization(Network* net);

#endif // AETHER_H
