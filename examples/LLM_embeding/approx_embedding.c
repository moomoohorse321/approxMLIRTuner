#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_DOCS 100000
#define EMBEDDING_DIM 384
#define MAX_LINE_LENGTH 100000

typedef struct {
    int id;
    float embedding[EMBEDDING_DIM];
    char title[256];
    float similarity;
} Document;

Document documents[MAX_DOCS];
int num_documents = 0;
float query_embedding[EMBEDDING_DIM];

/* -------------------- Prototypes for knobbed functions -------------------- */
/* exact + approximate implementation pairs (func_substitute target names)    */
float cosine_similarity_core(const float *a, const float *b, int state);
float approx_cosine_similarity_core(const float *a, const float *b, int state);

void rank_topk(int top_k, int state);
void approx_rank_topk(int top_k, int state);

/* -------------------- Helpers (unchanged I/O parsing) -------------------- */
int parse_embedding(const char *embedding_str, float *embedding) {
    const char *start = strchr(embedding_str, '[');
    if (!start) return 0;
    start++;

    int count = 0;
    char *copy = strdup(start);
    if (!copy) return 0;
    char *saveptr = NULL;

    char *token = strtok_r(copy, ",]", &saveptr);
    while (token && count < EMBEDDING_DIM) {
        embedding[count] = (float)atof(token);
        count++;
        token = strtok_r(NULL, ",]", &saveptr);
    }

    free(copy);
    return count == EMBEDDING_DIM;
}

/* -------------------- JSON loader (replaces stdin loader) -------------------- */
/* Only handles JSON of the shape:
   { "documents": [ { "title": "...", "content": "...", "embedding": [ ... ] }, ... ] }
   Minimal, schema-focused parsing (no external deps). */
int load_document_embeddings(const char *json_file_path) {
    FILE *fp = fopen(json_file_path, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Failed to open JSON file: %s\n", json_file_path);
        return 0;
    }

    /* read whole file */
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return 0; }
    long fsz = ftell(fp);
    if (fsz < 0) { fclose(fp); return 0; }
    if (fseek(fp, 0, SEEK_SET) != 0) { fclose(fp); return 0; }

    char *buf = (char *)malloc((size_t)fsz + 1);
    if (!buf) { fclose(fp); return 0; }
    size_t rd = fread(buf, 1, (size_t)fsz, fp);
    fclose(fp);
    buf[rd] = '\0';

    const char *p = buf;
    num_documents = 0;

    while (num_documents < MAX_DOCS) {
        /* find "title" */
        const char *tkey = strstr(p, "\"title\"");
        if (!tkey) break;
        const char *tcolon = strchr(tkey, ':');
        if (!tcolon) break;
        const char *q1 = strchr(tcolon, '\"');
        if (!q1) break;
        q1++;
        const char *q2 = strchr(q1, '\"');
        if (!q2) break;

        /* copy title (truncate to 255) */
        size_t tlen = (size_t)(q2 - q1);
        size_t cpy = tlen > 255 ? 255 : tlen;
        memcpy(documents[num_documents].title, q1, cpy);
        documents[num_documents].title[cpy] = '\0';

        /* find "embedding" after the title */
        const char *ekey = strstr(q2, "\"embedding\"");
        if (!ekey) break;
        const char *br1 = strchr(ekey, '[');
        if (!br1) break;
        const char *br2 = strchr(br1, ']');
        if (!br2) break;

        /* make a bounded substring "[ ... ]" for parse_embedding */
        size_t elen = (size_t)(br2 - br1 + 1);
        char *emb_str = (char *)malloc(elen + 1);
        if (!emb_str) { free(buf); return num_documents; }
        memcpy(emb_str, br1, elen);
        emb_str[elen] = '\0';

        if (parse_embedding(emb_str, documents[num_documents].embedding)) {
            documents[num_documents].id = num_documents; /* synthesize id: 0,1,2,... */
            num_documents++;
        }
        free(emb_str);

        /* continue scanning after this embedding */
        p = br2 + 1;
    }

    free(buf);
    return num_documents;
}

/* -------------------- Knob 1: cosine similarity core -------------------- */
/* Exact version: full-dimension cosine similarity                          */
float cosine_similarity_core(const float *a, const float *b, int state) {
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (int i = 0; i < EMBEDDING_DIM; i++) {
        float ai = a[i];
        float bi = b[i];
        dot_product += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    norm_a = sqrtf(norm_a);
    norm_b = sqrtf(norm_b);

    if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
    return dot_product / (norm_a * norm_b);
}

/* Approx version: perforate dimensions by stride-2 sampling                */
/* (keeps behavior compatible; the pass will swap this in if chosen)        */
float approx_cosine_similarity_core(const float *a, const float *b, int state) {
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (int i = 0; i < EMBEDDING_DIM; i += 2) {
        float ai = a[i];
        float bi = b[i];
        dot_product += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    /* simple renormalization to roughly account for skipped dims */
    dot_product *= 2.0f;
    norm_a = sqrtf(norm_a * 2.0f);
    norm_b = sqrtf(norm_b * 2.0f);

    if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
    return dot_product / (norm_a * norm_b);
}

/* Computes similarities using the chosen core (call site decides state). */
void compute_similarities_with_state(int cos_state) {
    for (int i = 0; i < num_documents; i++) {
        documents[i].similarity =
            cosine_similarity_core(query_embedding, documents[i].embedding, cos_state);
    }
}

/* -------------------- Sorting baseline (compare) -------------------- */
int compare_docs_desc(const void *a, const void *b) {
    const Document *da = (const Document *)a;
    const Document *db = (const Document *)b;
    if (da->similarity > db->similarity) return -1;
    if (da->similarity < db->similarity) return 1;
    return 0;
}

/* -------------------- Knob 2: rank top-k ----------------------------- */
/* Exact version: full qsort of all docs                                 */
void rank_topk(int top_k, int state) {
    (void)state; /* not used here; state is for the knob */
    qsort(documents, num_documents, sizeof(Document), compare_docs_desc);
}

/* Approx version: single-pass partial selection keeping only top_k       */
/* Places top_k best docs in documents[0..top_k-1]; rest left as-is.      */
void approx_rank_topk(int top_k, int state) {
    if (top_k <= 0) return;
    if (top_k > num_documents) top_k = num_documents;

    Document *buf = (Document *)malloc(sizeof(Document) * (size_t)top_k);
    int sz = 0;

    for (int i = 0; i < num_documents; i++) {
        Document cur = documents[i];

        /* insert into buf (descending by similarity) */
        int pos = sz;
        while (pos > 0 && buf[pos - 1].similarity < cur.similarity) {
            if (pos < top_k) buf[pos] = buf[pos - 1];
            pos--;
        }
        if (sz < top_k) {
            /* make room if shifting occurred above */
            if (pos < sz) buf[pos] = cur;
            else buf[sz] = cur;
            sz++;
        } else if (pos < top_k) {
            /* replace the last if improved */
            buf[top_k - 1] = cur;
            /* local bubble to restore order among last few */
            for (int j = top_k - 1; j > 0 && buf[j - 1].similarity < buf[j].similarity; --j) {
                Document tmp = buf[j - 1];
                buf[j - 1] = buf[j];
                buf[j] = tmp;
            }
        }
    }

    /* copy top_k back to the front */
    for (int i = 0; i < top_k; i++) documents[i] = buf[i];
    free(buf);
}

/* Printing (unchanged), now assumes rank_* already ran */
void output_top_k(int top_k) {
    int results = (top_k < num_documents) ? top_k : num_documents;
    for (int i = 0; i < results; i++) {
        printf("Rank %d: Doc %d (Score: %.4f) - \"%s\"\n",
               i + 1, documents[i].id, documents[i].similarity, documents[i].title);
    }
}

/* -------------------- Main -------------------- */
int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <query_embedding> <json_file_path> [top_k]\n", argv[0]);
        fprintf(stderr, "Query embedding should be comma-separated values\n");
        fprintf(stderr, "Documents must be provided via a JSON file (no stdin)\n");
        return 1;
    }

    const char *query_embedding_str = argv[1];
    const char *json_file_path = argv[2];
    int top_k = (argc > 3) ? atoi(argv[3]) : 10;

    char formatted_query[strlen(query_embedding_str) + 3];
    sprintf(formatted_query, "[%s]", query_embedding_str);

    if (!parse_embedding(formatted_query, query_embedding)) {
        fprintf(stderr, "Error: Failed to parse query embedding\n");
        return 1;
    }

    if (load_document_embeddings(json_file_path) == 0) {
        fprintf(stderr, "Error: No documents loaded from JSON\n");
        return 1;
    }

    printf("Loaded %d documents\n", num_documents);

    /* Decide states (can be constant or data-driven). Here: constant. */
    int cos_state = 0;   /* 0 → exact, tuner may choose approx via func_substitute */
    int rank_state = 0;  /* 0 → exact, tuner may choose approx via func_substitute */

    compute_similarities_with_state(cos_state);
    rank_topk(top_k, rank_state);
    output_top_k(top_k);

    return 0;
}