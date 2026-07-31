#include "aether_core.h"
#include <stdlib.h>
#include <math.h>

CROFLayer* crof_layer_create(int d_model, int d_state, float tau) {
    CROFLayer* layer = (CROFLayer*)malloc(sizeof(CROFLayer));
    layer->d_model = d_model;
    layer->d_state = d_state;
    int d_ff = ((int)(8.0f / 3.0f * d_model) + 63) / 64 * 64;
    layer->d_ff = d_ff;
    layer->tau = tau;

    layer->conv1d_weight = tensor_create(d_model, 4);
    layer->conv1d_bias   = tensor_create(d_model, 1);

    layer->lam_re = tensor_create(d_state, 1);
    layer->lam_im = tensor_create(d_state, 1);
    layer->gamma  = tensor_create(d_state, 1);

    layer->B_proj = tensor_create(d_state, d_model);
    layer->C_re   = tensor_create(d_model, d_state);
    layer->C_im   = tensor_create(d_model, d_state);

    layer->f_net_W = tensor_create(d_model, d_model);
    layer->f_net_b = tensor_create(d_model, 1);
    layer->g_net_W = tensor_create(d_model, d_model);
    layer->g_net_b = tensor_create(d_model, 1);
    layer->h_net_W = tensor_create(d_model, d_model);
    layer->h_net_b = tensor_create(d_model, 1);

    layer->norm_weight = tensor_create(d_model, 1);

    layer->ffn_w1 = tensor_create(d_ff, d_model);
    layer->ffn_w2 = tensor_create(d_ff, d_model);
    layer->ffn_w3 = tensor_create(d_model, d_ff);
    layer->norm_ffn_weight = tensor_create(d_model, 1);

    layer->s_re = tensor_create(d_state, 1);
    layer->s_im = tensor_create(d_state, 1);
    layer->conv_buf = tensor_create(4, d_model);

    layer->buf_x_conv  = tensor_create(d_model, 1);
    layer->buf_u       = tensor_create(d_state, 1);
    layer->buf_new_re  = tensor_create(d_state, 1);
    layer->buf_new_im  = tensor_create(d_state, 1);
    layer->buf_mem_re  = tensor_create(d_model, 1);
    layer->buf_mem_im  = tensor_create(d_model, 1);
    layer->buf_mem    = tensor_create(d_model, 1);
    layer->buf_z      = tensor_create(d_model, 1);
    layer->buf_f      = tensor_create(d_model, 1);
    layer->buf_gate   = tensor_create(d_model, 1);
    layer->buf_g      = tensor_create(d_model, 1);
    layer->buf_h_act  = tensor_create(d_model, 1);
    layer->buf_h      = tensor_create(d_model, 1);
    layer->buf_norm_ffn= tensor_create(d_model, 1);
    layer->buf_w1      = tensor_create(d_ff, 1);
    layer->buf_w2      = tensor_create(d_ff, 1);
    layer->buf_ffn_out = tensor_create(d_model, 1);

    crof_layer_reset(layer);
    return layer;
}

void crof_layer_free(CROFLayer* layer) {
    if (!layer) return;
    tensor_free(layer->conv1d_weight);
    tensor_free(layer->conv1d_bias);
    tensor_free(layer->lam_re);
    tensor_free(layer->lam_im);
    tensor_free(layer->gamma);
    tensor_free(layer->B_proj);
    tensor_free(layer->C_re);
    tensor_free(layer->C_im);
    tensor_free(layer->f_net_W);
    tensor_free(layer->f_net_b);
    tensor_free(layer->g_net_W);
    tensor_free(layer->g_net_b);
    tensor_free(layer->h_net_W);
    tensor_free(layer->h_net_b);
    tensor_free(layer->norm_weight);
    tensor_free(layer->ffn_w1);
    tensor_free(layer->ffn_w2);
    tensor_free(layer->ffn_w3);
    tensor_free(layer->norm_ffn_weight);
    tensor_free(layer->s_re);
    tensor_free(layer->s_im);
    tensor_free(layer->conv_buf);
    tensor_free(layer->buf_x_conv);
    tensor_free(layer->buf_u);
    tensor_free(layer->buf_new_re);
    tensor_free(layer->buf_new_im);
    tensor_free(layer->buf_mem_re);
    tensor_free(layer->buf_mem_im);
    tensor_free(layer->buf_mem);
    tensor_free(layer->buf_z);
    tensor_free(layer->buf_f);
    tensor_free(layer->buf_gate);
    tensor_free(layer->buf_g);
    tensor_free(layer->buf_h_act);
    tensor_free(layer->buf_h);
    tensor_free(layer->buf_norm_ffn);
    tensor_free(layer->buf_w1);
    tensor_free(layer->buf_w2);
    tensor_free(layer->buf_ffn_out);
    free(layer);
}

void crof_layer_reset(CROFLayer* layer) {
    if (!layer) return;
    tensor_zero(layer->s_re);
    tensor_zero(layer->s_im);
    tensor_zero(layer->conv_buf);
}

void crof_layer_step(CROFLayer* layer, const Tensor* x_t, Tensor* out_t) {
    int S = layer->d_state;
    int H = layer->d_model;
    int F_dim = layer->d_ff;

    // 1. Update circular buffer for Conv1d (k=4)
    for (int k = 0; k < 3; k++) {
        for (int j = 0; j < H; j++) {
            layer->conv_buf->data[k * H + j] = layer->conv_buf->data[(k + 1) * H + j];
        }
    }
    for (int j = 0; j < H; j++) {
        layer->conv_buf->data[3 * H + j] = x_t->data[j];
    }

    // Conv1d output per channel: y[t] = sum_k w[k] * x[t-3+k] + bias
    // conv_buf[k] = x[t-3+k] (buf[3] = token actual), identico al forward de PyTorch
    // (Conv1d con padding=3 truncada: el token actual multiplica w[3])
    for (int j = 0; j < H; j++) {
        float val = layer->conv1d_bias->data[j];
        for (int k = 0; k < 4; k++) {
            val += layer->conv_buf->data[k * H + j] * layer->conv1d_weight->data[j * 4 + k];
        }
        // SiLU
        float sig = 1.0f / (1.0f + expf(-val));
        layer->buf_x_conv->data[j] = val * sig;
    }

    // 2. u = B_proj * x_conv [S, 1] * gamma [S, 1]
    tensor_matmul(layer->buf_u, layer->B_proj, layer->buf_x_conv);
    for (int i = 0; i < S; i++) {
        layer->buf_u->data[i] *= layer->gamma->data[i];
    }

    // 3. Complex Oscillatory Recurrence
    for (int i = 0; i < S; i++) {
        float l_re = layer->lam_re->data[i];
        float l_im = layer->lam_im->data[i];
        float sre  = layer->s_re->data[i];
        float sim  = layer->s_im->data[i];
        float u_i  = layer->buf_u->data[i];

        layer->buf_new_re->data[i] = l_re * sre - l_im * sim + u_i;
        layer->buf_new_im->data[i] = l_re * sim + l_im * sre;
    }
    tensor_copy(layer->s_re, layer->buf_new_re);
    tensor_copy(layer->s_im, layer->buf_new_im);

    // 4. mem = C_re * s_re + C_im * s_im
    tensor_matmul(layer->buf_mem_re, layer->C_re, layer->s_re);
    tensor_matmul(layer->buf_mem_im, layer->C_im, layer->s_im);
    tensor_add(layer->buf_mem, layer->buf_mem_re, layer->buf_mem_im);

    // 5. z = x_conv + mem
    tensor_add(layer->buf_z, layer->buf_x_conv, layer->buf_mem);

    // 6. gate = sigmoid(-softplus(f_net_W * z + f_net_b) * tau)
    tensor_matmul(layer->buf_f, layer->f_net_W, layer->buf_z);
    tensor_add(layer->buf_f, layer->buf_f, layer->f_net_b);
    tensor_softplus(layer->buf_f, layer->buf_f);
    for (int i = 0; i < H; i++) {
        layer->buf_f->data[i] = -layer->buf_f->data[i] * layer->tau;
    }
    tensor_sigmoid(layer->buf_gate, layer->buf_f);

    // 7. g = g_net_W * z + g_net_b
    tensor_matmul(layer->buf_g, layer->g_net_W, layer->buf_z);
    tensor_add(layer->buf_g, layer->buf_g, layer->g_net_b);

    // 8. h_act = tanh(h_net_W * z + h_net_b)
    tensor_matmul(layer->buf_h_act, layer->h_net_W, layer->buf_z);
    tensor_add(layer->buf_h_act, layer->buf_h_act, layer->h_net_b);
    tensor_tanh(layer->buf_h_act, layer->buf_h_act);

    // 9. h = gate * g + (1 - gate) * h_act
    for (int i = 0; i < H; i++) {
        float g_val = layer->buf_gate->data[i];
        layer->buf_h->data[i] = g_val * layer->buf_g->data[i] + (1.0f - g_val) * layer->buf_h_act->data[i];
    }

    // 10. x_crof = RMSNorm(x_t + h)
    tensor_add(layer->buf_h, x_t, layer->buf_h);
    tensor_rmsnorm(layer->buf_h, layer->buf_h, layer->norm_weight, 1e-6f);

    // 11. SwiGLU FFN: out = x_crof + w3( silu(w1(norm(x_crof))) * w2(norm(x_crof)) )
    tensor_rmsnorm(layer->buf_norm_ffn, layer->buf_h, layer->norm_ffn_weight, 1e-6f);
    tensor_matmul(layer->buf_w1, layer->ffn_w1, layer->buf_norm_ffn);
    tensor_silu(layer->buf_w1, layer->buf_w1);
    tensor_matmul(layer->buf_w2, layer->ffn_w2, layer->buf_norm_ffn);
    for (int i = 0; i < F_dim; i++) {
        layer->buf_w1->data[i] *= layer->buf_w2->data[i];
    }
    tensor_matmul(layer->buf_ffn_out, layer->ffn_w3, layer->buf_w1);

    // Final residual output
    tensor_add(out_t, layer->buf_h, layer->buf_ffn_out);
}
