#include "aether_core.h"
#include <stdlib.h>
#include <stdio.h>

EWC_Params* ewc_create(Tensor* target_params) {
    if (!target_params) return NULL;

    EWC_Params* ewc = (EWC_Params*)malloc(sizeof(EWC_Params));
    ewc->params = target_params; // Pointer to the live weights

    ewc->fisher = tensor_create(target_params->rows, target_params->cols);
    ewc->anchor = tensor_create(target_params->rows, target_params->cols);

    tensor_zero(ewc->fisher);
    tensor_zero(ewc->anchor);

    return ewc;
}

void ewc_free(EWC_Params* ewc) {
    if (!ewc) return;
    tensor_free(ewc->fisher);
    tensor_free(ewc->anchor);
    free(ewc);
}

void ewc_snapshot(EWC_Params* ewc) {
    if (!ewc || !ewc->params || !ewc->anchor) return;

    // Save current parameters to anchor
    tensor_copy(ewc->anchor, ewc->params);

    // In Phase 0/1, we just simulate uniform Fisher information.
    // In later phases, this will be populated via accumulated gradients.
    int size = ewc->fisher->rows * ewc->fisher->cols;
    for(int i = 0; i < size; i++) {
        ewc->fisher->data[i] = 1.0f; // Uniform penalty
    }
}

// loss += ewc_penalty = (lambda / 2) * sum(Fisher * (params - anchor)^2)
float ewc_penalty(EWC_Params* ewc, float lambda) {
    if (!ewc || !ewc->params || !ewc->anchor || !ewc->fisher) return 0.0f;

    float penalty = 0.0f;
    int size = ewc->params->rows * ewc->params->cols;

    for (int i = 0; i < size; i++) {
        float diff = ewc->params->data[i] - ewc->anchor->data[i];
        penalty += ewc->fisher->data[i] * diff * diff;
    }

    return (lambda / 2.0f) * penalty;
}
