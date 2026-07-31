#include "aether_core.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>

Sampler* sampler_create(int vocab_size, float temperature, float top_p) {
    Sampler* s = (Sampler*)malloc(sizeof(Sampler));
    s->vocab_size = vocab_size;
    s->temperature = temperature;
    s->top_p = top_p;
    s->probs = (float*)calloc(vocab_size, sizeof(float));
    s->sorted = (ProbIndex*)calloc(vocab_size, sizeof(ProbIndex));
    return s;
}

void sampler_free(Sampler* s) {
    if (!s) return;
    if (s->probs) free(s->probs);
    if (s->sorted) free(s->sorted);
    free(s);
}

int sampler_argmax(const Tensor* logits) {
    if (!logits || !logits->data) return 0;
    int best = 0;
    float best_val = logits->data[0];
    int size = logits->rows * logits->cols;
    for (int i = 1; i < size; i++) {
        if (logits->data[i] > best_val) {
            best_val = logits->data[i];
            best = i;
        }
    }
    return best;
}

static int cmp_prob_desc(const void* a, const void* b) {
    float pa = ((const ProbIndex*)a)->prob;
    float pb = ((const ProbIndex*)b)->prob;
    if (pa > pb) return -1;
    if (pa < pb) return 1;
    return 0;
}

int sampler_sample(Sampler* s, const Tensor* logits) {
    if (!s || !logits || !logits->data) return 0;

    if (s->temperature <= 0.0f) {
        return sampler_argmax(logits);
    }

    int n = s->vocab_size;

    // Softmax con temperatura (numericamente estable)
    float max_val = logits->data[0];
    for (int i = 1; i < n; i++) {
        if (logits->data[i] > max_val) max_val = logits->data[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        s->probs[i] = expf((logits->data[i] - max_val) / s->temperature);
        sum += s->probs[i];
    }
    for (int i = 0; i < n; i++) {
        s->probs[i] /= sum;
        s->sorted[i].prob = s->probs[i];
        s->sorted[i].index = i;
    }

    // Top-P (nucleus): ordenar desc y cortar en masa acumulada >= top_p
    qsort(s->sorted, (size_t)n, sizeof(ProbIndex), cmp_prob_desc);

    int kept = n;
    if (s->top_p > 0.0f && s->top_p < 1.0f) {
        float cumulative = 0.0f;
        for (int i = 0; i < n; i++) {
            cumulative += s->sorted[i].prob;
            if (cumulative >= s->top_p) {
                kept = i + 1;
                break;
            }
        }
    }

    // Renormalizar el nucleo y muestrear
    float kept_sum = 0.0f;
    for (int i = 0; i < kept; i++) kept_sum += s->sorted[i].prob;

    float r = ((float)rand() / (float)RAND_MAX) * kept_sum;
    float cumulative = 0.0f;
    for (int i = 0; i < kept; i++) {
        cumulative += s->sorted[i].prob;
        if (r <= cumulative) {
            return s->sorted[i].index;
        }
    }
    return s->sorted[kept - 1].index;
}
