#include "aether_core.h"
#include <stdlib.h>
#include <string.h>

EmbeddingTable* embedding_create(int vocab_size, int embed_dim) {
    EmbeddingTable* emb = (EmbeddingTable*)malloc(sizeof(EmbeddingTable));
    emb->vocab_size = vocab_size;
    emb->embed_dim = embed_dim;
    emb->weight = tensor_create(vocab_size, embed_dim);
    tensor_randomize(emb->weight, -0.1f, 0.1f);
    return emb;
}

void embedding_free(EmbeddingTable* emb) {
    if (!emb) return;
    tensor_free(emb->weight);
    free(emb);
}

// Look up token_id in the embedding table, write result to out [embed_dim, 1]
void embedding_lookup(const EmbeddingTable* emb, int token_id, Tensor* out) {
    if (!emb || !out) return;
    if (token_id < 0 || token_id >= emb->vocab_size) {
        tensor_zero(out);
        return;
    }
    // Row token_id of the weight matrix → column vector out
    for (int i = 0; i < emb->embed_dim; i++) {
        out->data[i] = emb->weight->data[token_id * emb->embed_dim + i];
    }
}
