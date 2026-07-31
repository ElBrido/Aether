#include "aether_core.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static int read_floats(FILE* fp, Tensor* t, const char* name) {
    int count = t->rows * t->cols;
    size_t read = fread(t->data, sizeof(float), count, fp);
    if ((int)read != count) {
        fprintf(stderr, "[LOADER ERROR] Failed to read '%s': expected %d floats, got %zu\n",
                name, count, read);
        return -1;
    }
    printf("  %-42s -> [%4d x %4d] (%d floats)\n", name, t->rows, t->cols, count);
    return 0;
}

AetherEngine* aether_load_from_bin(const char* filepath) {
    printf("[LOADER] Abriendo archivo binario CROF v3.0: %s\n", filepath);

    FILE* fp = fopen(filepath, "rb");
    if (!fp) {
        fprintf(stderr, "[LOADER ERROR] No se pudo abrir '%s'\n", filepath);
        return NULL;
    }

    AetherHeader header;
    if (fread(&header, sizeof(AetherHeader), 1, fp) != 1) {
        fprintf(stderr, "[LOADER ERROR] No se pudo leer el header\n");
        fclose(fp);
        return NULL;
    }

    if (header.magic != AETHER_MAGIC) {
        fprintf(stderr, "[LOADER ERROR] Magic number inválido: 0x%08X (esperado 0x%08X)\n",
                header.magic, AETHER_MAGIC);
        fclose(fp);
        return NULL;
    }

    if (header.version != AETHER_VERSION && header.version != 1) {
        fprintf(stderr, "[LOADER ERROR] Versión incompatible: %u (esperada %u o 1)\n",
                header.version, AETHER_VERSION);
        fclose(fp);
        return NULL;
    }

    printf("[LOADER] Header válido (CROF v3.0 Multi-Layer):\n");
    printf("  vocab_size    = %u\n", header.vocab_size);
    printf("  hidden_dim    = %u\n", header.hidden_dim);
    printf("  ssm_state_dim = %u\n", header.ssm_state_dim);
    printf("  num_layers    = %u\n", header.num_layers);
    printf("  chaos_sigma   = %.6f\n", header.chaos_sigma);

    int V = (int)header.vocab_size;
    int H = (int)header.hidden_dim;
    int S = (int)header.ssm_state_dim;
    int L = (int)header.num_layers;

    AetherEngine* engine = aether_create(V, H, S, L);
    engine->chaos_sigma = header.chaos_sigma;

    printf("\n[LOADER] Cargando pesos CROF v3.0 (%d bloques)...\n", L);

    int err = 0;

    // Embedding y normas iniciales (RMSNorm)
    err |= read_floats(fp, engine->embedding->weight, "embedding.weight");
    err |= read_floats(fp, engine->pos_norm_weight, "pos_norm.weight");

    // Cargar cada uno de los num_layers bloques CROF
    for (int l = 0; l < L; l++) {
        char name[128];
        CROFLayer* block = engine->crof_layers[l];

        snprintf(name, sizeof(name), "blocks.%d.conv1d.weight", l);
        err |= read_floats(fp, block->conv1d_weight, name);
        snprintf(name, sizeof(name), "blocks.%d.conv1d.bias", l);
        err |= read_floats(fp, block->conv1d_bias, name);

        snprintf(name, sizeof(name), "blocks.%d.lam_re", l);
        err |= read_floats(fp, block->lam_re, name);
        snprintf(name, sizeof(name), "blocks.%d.lam_im", l);
        err |= read_floats(fp, block->lam_im, name);
        snprintf(name, sizeof(name), "blocks.%d.gamma", l);
        err |= read_floats(fp, block->gamma, name);

        snprintf(name, sizeof(name), "blocks.%d.B_proj.weight", l);
        err |= read_floats(fp, block->B_proj, name);
        snprintf(name, sizeof(name), "blocks.%d.C_re.weight", l);
        err |= read_floats(fp, block->C_re, name);
        snprintf(name, sizeof(name), "blocks.%d.C_im.weight", l);
        err |= read_floats(fp, block->C_im, name);

        snprintf(name, sizeof(name), "blocks.%d.f_net.weight", l);
        err |= read_floats(fp, block->f_net_W, name);
        snprintf(name, sizeof(name), "blocks.%d.f_net.bias", l);
        err |= read_floats(fp, block->f_net_b, name);
        snprintf(name, sizeof(name), "blocks.%d.g_net.weight", l);
        err |= read_floats(fp, block->g_net_W, name);
        snprintf(name, sizeof(name), "blocks.%d.g_net.bias", l);
        err |= read_floats(fp, block->g_net_b, name);
        snprintf(name, sizeof(name), "blocks.%d.h_net.weight", l);
        err |= read_floats(fp, block->h_net_W, name);
        snprintf(name, sizeof(name), "blocks.%d.h_net.bias", l);
        err |= read_floats(fp, block->h_net_b, name);

        snprintf(name, sizeof(name), "blocks.%d.norm.weight", l);
        err |= read_floats(fp, block->norm_weight, name);

        snprintf(name, sizeof(name), "blocks.%d.ffn.w1.weight", l);
        err |= read_floats(fp, block->ffn_w1, name);
        snprintf(name, sizeof(name), "blocks.%d.ffn.w2.weight", l);
        err |= read_floats(fp, block->ffn_w2, name);
        snprintf(name, sizeof(name), "blocks.%d.ffn.w3.weight", l);
        err |= read_floats(fp, block->ffn_w3, name);
        snprintf(name, sizeof(name), "blocks.%d.norm_ffn.weight", l);
        err |= read_floats(fp, block->norm_ffn_weight, name);

        float tau_val;
        if (fread(&tau_val, sizeof(float), 1, fp) != 1) {
            fprintf(stderr, "[LOADER ERROR] Failed to read tau for block %d\n", l);
            err = -1;
        } else {
            block->tau = tau_val;
            printf("  blocks.%-33d.tau -> scalar = %.6f\n", l, tau_val);
        }
    }

    // Norm final y cabezal de salida
    err |= read_floats(fp, engine->final_norm_weight, "final_norm.weight");
    err |= read_floats(fp, engine->W_out, "fc_out.weight");
    err |= read_floats(fp, engine->b_out, "fc_out.bias");

    fclose(fp);

    if (err != 0) {
        fprintf(stderr, "[LOADER ERROR] Errores durante la carga. Modelo incompleto.\n");
        aether_free(engine);
        return NULL;
    }

    printf("\n[LOADER] Pesos CROF v3.0 (%d capas) cargados exitosamente. Motor A.E.T.H.E.R. listo.\n", L);
    return engine;
}
