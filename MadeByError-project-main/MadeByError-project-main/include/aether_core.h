#ifndef AETHER_CORE_H
#define AETHER_CORE_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

// ---------------------------------------------------------
// TENSOR SYSTEM (Pure C, No Dependencies)
// ---------------------------------------------------------
typedef struct Tensor {
    int rows;
    int cols;
    float* data; // Contiguous 1D array for 2D data (using float for performance)
} Tensor;

Tensor* tensor_create(int rows, int cols);
void tensor_free(Tensor* t);
void tensor_randomize(Tensor* t, float min_val, float max_val);
void tensor_zero(Tensor* t);
void tensor_add(Tensor* out, const Tensor* a, const Tensor* b);
void tensor_matmul(Tensor* out, const Tensor* a, const Tensor* b);
void tensor_scale(Tensor* t, float scalar);
void tensor_copy(Tensor* dst, const Tensor* src);
void tensor_print(const Tensor* t, const char* name, int max_elems);

// RMSNorm & Activation functions
void tensor_rmsnorm(Tensor* out, const Tensor* in, const Tensor* weight, float eps);
void tensor_silu(Tensor* out, const Tensor* in);
void tensor_sigmoid(Tensor* out, const Tensor* in);
void tensor_tanh(Tensor* out, const Tensor* in);
void tensor_softplus(Tensor* out, const Tensor* in);

// ---------------------------------------------------------
// CROF LAYER v3.0 (Closed-form Rotational Oscillatory Field)
// ---------------------------------------------------------
typedef struct CROFLayer {
    int d_model;
    int d_state;
    int d_ff;
    float tau;

    // Conv1d Causal corta (k=4)
    Tensor* conv1d_weight; // [d_model, 4]
    Tensor* conv1d_bias;   // [d_model, 1]

    // Precomputed lambda parameters (exported from PyTorch _lambda())
    Tensor* lam_re;   // [d_state, 1]
    Tensor* lam_im;   // [d_state, 1]
    Tensor* gamma;    // [d_state, 1]

    Tensor* B_proj;   // [d_state, d_model]
    Tensor* C_re;     // [d_model, d_state]
    Tensor* C_im;     // [d_model, d_state]

    Tensor* f_net_W;  // [d_model, d_model]
    Tensor* f_net_b;  // [d_model, 1]
    Tensor* g_net_W;  // [d_model, d_model]
    Tensor* g_net_b;  // [d_model, 1]
    Tensor* h_net_W;  // [d_model, d_model]
    Tensor* h_net_b;  // [d_model, 1]

    Tensor* norm_weight; // [d_model, 1] RMSNorm weight

    // SwiGLU FFN
    Tensor* ffn_w1;          // [d_ff, d_model]
    Tensor* ffn_w2;          // [d_ff, d_model]
    Tensor* ffn_w3;          // [d_model, d_ff]
    Tensor* norm_ffn_weight; // [d_model, 1] RMSNorm weight

    // Rotational Oscillatory State s_re, s_im (Complex Memory) + Conv1d circular buffer
    Tensor* s_re;     // [d_state, 1]
    Tensor* s_im;     // [d_state, 1]
    Tensor* conv_buf; // [4, d_model] circular buffer

    // Scratch Buffers
    Tensor* buf_x_conv;  // [d_model, 1]
    Tensor* buf_u;       // [d_state, 1]
    Tensor* buf_new_re;  // [d_state, 1]
    Tensor* buf_new_im;  // [d_state, 1]
    Tensor* buf_mem_re;  // [d_model, 1]
    Tensor* buf_mem_im;  // [d_model, 1]
    Tensor* buf_mem;     // [d_model, 1]
    Tensor* buf_z;       // [d_model, 1]
    Tensor* buf_f;       // [d_model, 1]
    Tensor* buf_gate;    // [d_model, 1]
    Tensor* buf_g;       // [d_model, 1]
    Tensor* buf_h_act;   // [d_model, 1]
    Tensor* buf_h;       // [d_model, 1]
    Tensor* buf_norm_ffn;// [d_model, 1]
    Tensor* buf_w1;      // [d_ff, 1]
    Tensor* buf_w2;      // [d_ff, 1]
    Tensor* buf_ffn_out; // [d_model, 1]
} CROFLayer;

CROFLayer* crof_layer_create(int d_model, int d_state, float tau);
void crof_layer_free(CROFLayer* layer);
void crof_layer_reset(CROFLayer* layer);
void crof_layer_step(CROFLayer* layer, const Tensor* x_t, Tensor* out_t);

// ---------------------------------------------------------
// ELASTIC WEIGHT CONSOLIDATION (EWC)
// ---------------------------------------------------------
typedef struct EWC_Params {
    Tensor* params;
    Tensor* fisher;
    Tensor* anchor;
} EWC_Params;

EWC_Params* ewc_create(Tensor* target_params);
void ewc_free(EWC_Params* ewc);
void ewc_snapshot(EWC_Params* ewc);
float ewc_penalty(EWC_Params* ewc, float lambda);

// ---------------------------------------------------------
// EMBEDDING TABLE (token_id -> vector)
// ---------------------------------------------------------
typedef struct EmbeddingTable {
    int vocab_size;
    int embed_dim;
    Tensor* weight; // [vocab_size, embed_dim] row-major
} EmbeddingTable;

EmbeddingTable* embedding_create(int vocab_size, int embed_dim);
void embedding_free(EmbeddingTable* emb);
void embedding_lookup(const EmbeddingTable* emb, int token_id, Tensor* out);

// ---------------------------------------------------------
// BPE TOKENIZER (kairos_tokenizer.bin exportado desde PyTorch)
// ---------------------------------------------------------

#define AETK_MAGIC 0x4145544B  // 'AETK'

typedef struct BPETokenizer {
    int vocab_size;
    int num_merges;
    char** tokens;      // vocab_size cadenas UTF-8 (orden de id)
    int* merge_left;    // [num_merges]
    int* merge_right;   // [num_merges]
    int* merge_new;     // [num_merges]
    int pad_id, unk_id, bos_id, eos_id, sys_id, usr_id, ast_id;
} BPETokenizer;

BPETokenizer* bpe_load(const char* filepath);
void bpe_free(BPETokenizer* t);
int bpe_find_token(const BPETokenizer* t, const char* s);
int bpe_encode(const BPETokenizer* t, const char* text, int* out_ids, int max_ids);
void bpe_print_token(const BPETokenizer* t, int id);

// ---------------------------------------------------------
// MULTINOMIAL SAMPLER (with Temperature & Top-P)
// ---------------------------------------------------------

typedef struct ProbIndex {
    float prob;
    int index;
} ProbIndex;

typedef struct Sampler {
    float temperature;
    float top_p;
    float* probs;       // Scratch buffer [vocab_size]
    ProbIndex* sorted;  // Scratch buffer [vocab_size] para top-p
    int vocab_size;
} Sampler;

Sampler* sampler_create(int vocab_size, float temperature, float top_p);
void sampler_free(Sampler* s);
int sampler_sample(Sampler* s, const Tensor* logits);
int sampler_argmax(const Tensor* logits);

// ---------------------------------------------------------
// A.E.T.H.E.R. ENGINE v3.0 (Multi-Layer CROF Core)
// ---------------------------------------------------------
typedef struct AetherEngine {
    int vocab_size;
    int hidden_dim;
    int ssm_state_dim; // d_state
    int num_layers;
    float chaos_sigma;

    EmbeddingTable* embedding;
    Tensor* pos_norm_weight; // [hidden_dim, 1] RMSNorm

    CROFLayer** crof_layers; // Array of num_layers CROFLayer pointers

    Tensor* final_norm_weight; // [hidden_dim, 1] RMSNorm

    // Final linear projection to logits (Tied weights)
    Tensor* W_out;  // [vocab_size, hidden_dim]
    Tensor* b_out;  // [vocab_size, 1]

    // Buffers
    Tensor* logits;   // [vocab_size, 1]
    Tensor* emb_vec;  // [hidden_dim, 1]
    Tensor* norm_vec; // [hidden_dim, 1]
    Tensor* layer_in; // [hidden_dim, 1]
    Tensor* layer_out;// [hidden_dim, 1]
} AetherEngine;

AetherEngine* aether_create(int vocab_size, int hidden_dim, int ssm_state_dim, int num_layers);
void aether_free(AetherEngine* engine);
void aether_forward_token(AetherEngine* engine, int token_id);
void aether_reset(AetherEngine* engine);

// ---------------------------------------------------------
// BINARY WEIGHT LOADER (CROF v3.0 Multi-Layer format)
// ---------------------------------------------------------

#define AETHER_MAGIC 0x41455448  // 'AETH'
#define AETHER_VERSION 3         // CROF v3.0

typedef struct AetherHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t vocab_size;
    uint32_t hidden_dim;
    uint32_t ssm_state_dim;
    uint32_t num_layers;
    float    chaos_sigma;
} AetherHeader;

AetherEngine* aether_load_from_bin(const char* filepath);

#endif // AETHER_CORE_H
