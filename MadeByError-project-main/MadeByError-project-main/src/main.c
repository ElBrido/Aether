#include "aether_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char* SYSTEM_TEXT =
    "Eres Kairos, una Inteligencia Artificial hiper-logica creada por brido.";

#define MAX_PROMPT_IDS 4096

static void print_banner(void) {
    printf("\n");
    printf("  ==========================================================\n");
    printf("   A.E.T.H.E.R. - Motor de Inferencia v3.0 (CROF, C11)\n");
    printf("   Memoria Rotacional Compleja O(1) - Latencia Fija\n");
    printf("   BPE Espanol 16k - CPU Ultra-Bajo Costo - by brido\n");
    printf("  ==========================================================\n\n");
}

static void parity_test(AetherEngine* engine) {
    printf("=== TEST DE PARIDAD (Deterministico, Temperatura=0) ===\n");
    printf("Input: tokens [1, 1, 1]\n\n");

    aether_reset(engine);

    for (int i = 0; i < 3; i++) {
        aether_forward_token(engine, 1);
    }

    printf("Output Exacto (Primeros 20 floats del ultimo token):\n");
    int n = (engine->vocab_size < 20) ? engine->vocab_size : 20;
    for (int i = 0; i < n; i++) {
        printf("  [%02d]: %.6f\n", i, engine->logits->data[i]);
    }
    printf("\n");
}

static void generate_interactive(AetherEngine* engine, BPETokenizer* tok) {
    printf("=== MODO GENERACION INTERACTIVA (CROF v3.0 + BPE) ===\n");
    printf("Escribe tu prompt (o 'salir' para terminar).\n");
    printf("Temperatura: 0.8 | Top-P: 0.95 | Max tokens: 200\n\n");

    Sampler* sampler = sampler_create(engine->vocab_size, 0.8f, 0.95f);
    srand((unsigned int)time(NULL));

    char input_buf[1024];
    int prompt_ids[MAX_PROMPT_IDS];

    while (1) {
        printf(">> ");
        if (!fgets(input_buf, sizeof(input_buf), stdin)) break;

        size_t len = strlen(input_buf);
        if (len > 0 && input_buf[len - 1] == '\n') {
            input_buf[len - 1] = '\0';
            len--;
        }

        if (len == 0) continue;
        if (strcmp(input_buf, "salir") == 0) break;

        // Prompt identico al formato de entrenamiento:
        // <SYS> system_text <USR> user_text <AST>
        int n = 0;
        prompt_ids[n++] = tok->sys_id;
        n += bpe_encode(tok, SYSTEM_TEXT, prompt_ids + n, MAX_PROMPT_IDS - n - 2);
        prompt_ids[n++] = tok->usr_id;
        n += bpe_encode(tok, input_buf, prompt_ids + n, MAX_PROMPT_IDS - n - 1);
        prompt_ids[n++] = tok->ast_id;

        aether_reset(engine);
        for (int i = 0; i < n; i++) {
            aether_forward_token(engine, prompt_ids[i]);
        }

        printf("\nKairos: ");
        int max_tokens = 200;

        for (int t = 0; t < max_tokens; t++) {
            int next_tok = sampler_sample(sampler, engine->logits);

            if (next_tok == tok->eos_id) break;

            bpe_print_token(tok, next_tok);
            fflush(stdout);

            aether_forward_token(engine, next_tok);
        }
        printf("\n\n");
    }

    sampler_free(sampler);
}

int main(int argc, char* argv[]) {
    print_banner();

    const char* weights_path = "kairos_weights.bin";
    const char* tokenizer_path = "kairos_tokenizer.bin";
    if (argc > 1) weights_path = argv[1];
    if (argc > 2) tokenizer_path = argv[2];

    AetherEngine* engine = aether_load_from_bin(weights_path);

    if (engine) {
        printf("\n[Sistema] Motor CROF cargado desde pesos entrenados (%s).\n\n", weights_path);
    } else {
        printf("[Sistema] No se encontro '%s'. Usando pesos aleatorios (demo).\n\n", weights_path);
        engine = aether_create(16000, 512, 1024, 6);
    }

    parity_test(engine);

    BPETokenizer* tok = bpe_load(tokenizer_path);
    if (!tok) {
        printf("[Sistema] No se encontro el tokenizador '%s'.\n", tokenizer_path);
        printf("[Sistema] Exportalo desde PyTorch con export_tokenizer_bin() y vuelve a ejecutar.\n");
        aether_free(engine);
        return 1;
    }

    if (tok->vocab_size != engine->vocab_size) {
        printf("[Sistema] ADVERTENCIA: vocab del tokenizador (%d) != vocab del modelo (%d).\n",
               tok->vocab_size, engine->vocab_size);
    }

    generate_interactive(engine, tok);

    bpe_free(tok);
    aether_free(engine);
    printf("[Sistema] Motor apagado. Memoria liberada.\n");
    return 0;
}
