#include "aether_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------
// BPE TOKENIZER (formato binario kairos_tokenizer.bin)
// Header: u32 magic 'AETK' | u32 version | u32 vocab_size | u32 num_merges
// Vocab (en orden de id): u16 len | bytes UTF-8
// Merges (en orden de rank): u32 left_id | u32 right_id | u32 new_id
// ---------------------------------------------------------

static int utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

int bpe_find_token(const BPETokenizer* t, const char* s) {
    if (!t || !s) return -1;
    for (int i = 0; i < t->vocab_size; i++) {
        if (t->tokens[i] && strcmp(t->tokens[i], s) == 0) return i;
    }
    return -1;
}

BPETokenizer* bpe_load(const char* filepath) {
    FILE* fp = fopen(filepath, "rb");
    if (!fp) {
        fprintf(stderr, "[BPE ERROR] No se pudo abrir '%s'\n", filepath);
        return NULL;
    }

    uint32_t magic = 0, version = 0, vocab_size = 0, num_merges = 0;
    if (fread(&magic, 4, 1, fp) != 1 || fread(&version, 4, 1, fp) != 1 ||
        fread(&vocab_size, 4, 1, fp) != 1 || fread(&num_merges, 4, 1, fp) != 1) {
        fprintf(stderr, "[BPE ERROR] Header ilegible\n");
        fclose(fp);
        return NULL;
    }
    /* Aceptar AETK (v1) y BETK (v2 con tabla byte_to_id) */
    int has_byte_table = 0;
    if (magic == 0x4B544542u) { /* 'BETK' */
        has_byte_table = 1;
    } else if (magic != AETK_MAGIC) {
        fprintf(stderr, "[BPE ERROR] Magic invalido: 0x%08X\n", magic);
        fclose(fp);
        return NULL;
    }

    BPETokenizer* t = (BPETokenizer*)calloc(1, sizeof(BPETokenizer));
    t->vocab_size = (int)vocab_size;
    t->num_merges = (int)num_merges;
    t->tokens = (char**)calloc(vocab_size, sizeof(char*));
    t->merge_left  = (int*)malloc(num_merges * sizeof(int));
    t->merge_right = (int*)malloc(num_merges * sizeof(int));
    t->merge_new   = (int*)malloc(num_merges * sizeof(int));

    /* Leer tabla byte_to_id si es BETK v2 */
    if (has_byte_table) {
        for (int i = 0; i < 256; i++) {
            uint32_t bid = 0;
            if (fread(&bid, 4, 1, fp) != 1) { fprintf(stderr, "[BPE ERROR] byte_to_id truncado\n"); goto fail; }
            t->byte_to_id[i] = (int)bid;
        }
    } else {
        /* Fallback: Latin-1 byte -> 7+byte */
        for (int i = 0; i < 256; i++) t->byte_to_id[i] = 7 + i;
    }

    for (uint32_t i = 0; i < vocab_size; i++) {
        uint16_t len = 0;
        if (fread(&len, 2, 1, fp) != 1) { fprintf(stderr, "[BPE ERROR] Vocab truncado en %u\n", i); goto fail; }
        t->tokens[i] = (char*)malloc((size_t)len + 1);
        if (len > 0 && fread(t->tokens[i], 1, len, fp) != len) { fprintf(stderr, "[BPE ERROR] Token %u truncado\n", i); goto fail; }
        t->tokens[i][len] = '\0';
    }

    for (uint32_t i = 0; i < num_merges; i++) {
        uint32_t a = 0, b = 0, c = 0;
        if (fread(&a, 4, 1, fp) != 1 || fread(&b, 4, 1, fp) != 1 || fread(&c, 4, 1, fp) != 1) {
            fprintf(stderr, "[BPE ERROR] Merges truncados en %u\n", i);
            goto fail;
        }
        t->merge_left[i] = (int)a;
        t->merge_right[i] = (int)b;
        t->merge_new[i] = (int)c;
    }
    fclose(fp);

    t->pad_id = bpe_find_token(t, "<PAD>"); if (t->pad_id < 0) t->pad_id = 0;
    t->unk_id = bpe_find_token(t, "<UNK>"); if (t->unk_id < 0) t->unk_id = 1;
    t->bos_id = bpe_find_token(t, "<BOS>"); if (t->bos_id < 0) t->bos_id = 2;
    t->eos_id = bpe_find_token(t, "<EOS>"); if (t->eos_id < 0) t->eos_id = 3;
    t->sys_id = bpe_find_token(t, "<SYS>"); if (t->sys_id < 0) t->sys_id = 4;
    t->usr_id = bpe_find_token(t, "<USR>"); if (t->usr_id < 0) t->usr_id = 5;
    t->ast_id = bpe_find_token(t, "<AST>"); if (t->ast_id < 0) t->ast_id = 6;


    printf("[BPE] Tokenizador cargado: %d tokens, %d merges\n", t->vocab_size, t->num_merges);
    return t;

fail:
    fclose(fp);
    bpe_free(t);
    return NULL;
}

void bpe_free(BPETokenizer* t) {
    if (!t) return;
    if (t->tokens) {
        for (int i = 0; i < t->vocab_size; i++) free(t->tokens[i]);
        free(t->tokens);
    }
    free(t->merge_left);
    free(t->merge_right);
    free(t->merge_new);
    free(t);
}

static int find_merge_rank(const BPETokenizer* t, int a, int b) {
    for (int i = 0; i < t->num_merges; i++) {
        if (t->merge_left[i] == a && t->merge_right[i] == b) return i;
    }
    return -1;
}

#define BPE_MAX_WORD_SYMS 256

static int encode_word(const BPETokenizer* t, const char* word, int* out, int max_out) {
    int sym[BPE_MAX_WORD_SYMS];
    int n = 0;
    const char* p = word;

    // Simbolos iniciales: codepoints UTF-8
    while (*p && n < BPE_MAX_WORD_SYMS) {
        int l = utf8_len((unsigned char)*p);
        char buf[8];
        memcpy(buf, p, (size_t)l);
        buf[l] = '\0';
        int id = bpe_find_token(t, buf);
        sym[n++] = (id >= 0) ? id : t->unk_id;
        p += l;
    }

    // Fusionar siempre el par presente con menor rank (BPE estandar)
    while (n > 1) {
        int best_rank = -1, best_pos = -1;
        for (int i = 0; i < n - 1; i++) {
            int r = find_merge_rank(t, sym[i], sym[i + 1]);
            if (r >= 0 && (best_rank < 0 || r < best_rank)) {
                best_rank = r;
                best_pos = i;
            }
        }
        if (best_pos < 0) break;
        sym[best_pos] = t->merge_new[best_rank];
        for (int i = best_pos + 1; i < n - 1; i++) sym[i] = sym[i + 1];
        n--;
    }

    int c = 0;
    for (int i = 0; i < n && c < max_out; i++) out[c++] = sym[i];
    return c;
}

static int is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

int bpe_encode(const BPETokenizer* t, const char* text, int* out_ids, int max_ids) {
    if (!t || !text || !out_ids || max_ids <= 0) return 0;
    int len = (int)strlen(text);
    if (len == 0) return 0;

    int sym[4096];
    int n = 0;

    /* Prepend U+2581 (0xE2 0x96 0x81) if text does not start with space */
    if (text[0] != ' ' && text[0] != '\n' && text[0] != '\r' && text[0] != '\t') {
        sym[n++] = t->byte_to_id[0xE2];
        sym[n++] = t->byte_to_id[0x96];
        sym[n++] = t->byte_to_id[0x81];
    }

    for (int i = 0; i < len && n < 4090; i++) {
        unsigned char b = (unsigned char)text[i];
        if (b == ' ') {
            sym[n++] = t->byte_to_id[0xE2];
            sym[n++] = t->byte_to_id[0x96];
            sym[n++] = t->byte_to_id[0x81];
        } else {
            sym[n++] = t->byte_to_id[(int)b];
        }
    }

    /* BPE Merge ranking loop (greedy lowest rank) */
    while (n > 1) {
        int best_rank = -1, best_pos = -1;
        for (int i = 0; i < n - 1; i++) {
            int r = find_merge_rank(t, sym[i], sym[i + 1]);
            if (r >= 0 && (best_rank < 0 || r < best_rank)) {
                best_rank = r;
                best_pos = i;
            }
        }
        if (best_pos < 0) break;
        sym[best_pos] = t->merge_new[best_rank];
        for (int i = best_pos + 1; i < n - 1; i++) sym[i] = sym[i + 1];
        n--;
    }

    int count = (n < max_ids) ? n : max_ids;
    for (int i = 0; i < count; i++) out_ids[i] = sym[i];
    return count;
}

void bpe_print_token(const BPETokenizer* t, int id) {
    if (!t || id < 0 || id >= t->vocab_size) return;
    if (id == t->pad_id || id == t->unk_id || id == t->bos_id || id == t->eos_id ||
        id == t->sys_id || id == t->usr_id || id == t->ast_id) return;

    if (!t->tokens[id]) return;
    /* Reemplazar U+2581 (0xE2 0x96 0x81) por espacio al imprimir */
    const unsigned char* s = (const unsigned char*)t->tokens[id];
    size_t len = strlen((const char*)s);
    for (size_t i = 0; i < len; ) {
        if (i + 2 < len && s[i] == 0xE2 && s[i+1] == 0x96 && s[i+2] == 0x81) {
            putchar(' ');
            i += 3;
        } else {
            putchar((int)s[i]);
            i++;
        }
    }
}
