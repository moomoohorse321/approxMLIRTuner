// RUN: cgeist -O0 %stdinclude %s -S > %s.mlir

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>   // for strcasecmp
#include <ctype.h>
#include <math.h>
#include <time.h>

// --- BM25 Parameters ---
const double K1 = 1.5;
const double B  = 0.75;

// --- Helper Struct for Ranking ---
typedef struct {
    int doc_index;
    double score;
} DocumentScore;

// ---------- small helpers (no knobs) ----------
static inline char *lower_dup(const char *s) {
    if (!s) return NULL;
    size_t n = strlen(s);
    char *p = (char*)malloc(n+1);
    if (!p) return NULL;
    for (size_t i=0;i<n;i++) p[i] = (char)tolower((unsigned char)s[i]);
    p[n] = '\0';
    return p;
}

// case-insensitive token compare
int compare_tokens(const char *t1, const char *t2) {
    return strcasecmp(t1, t2);
}

// counts words and lowercases in-place
int count_and_lower_words(char *str) {
    if (str == NULL || *str == '\0') return 0;
    int count = 0;
    char *p = str;
    int in_word = 0;
    while (*p) {
        *p = (char)tolower((unsigned char)*p);
        if (isalnum((unsigned char)*p)) {
            if (!in_word) { in_word = 1; count++; }
        } else {
            in_word = 0;
        }
        p++;
    }
    return count;
}

double calculate_idf(int df, int num_docs) {
    double N = (double)num_docs;
    double n_q = (double)df;
    double idf = log(((N - n_q + 0.5) / (n_q + 0.5)) + 1.0);
    return (idf > 0.0) ? idf : 0.0;
}

// ---------- KNOB 1: TF counting (func_substitute) ----------
// Exact version (do not use 'state' here; it's caller-only or for nested knobs)
int tf_count_whole_word(const char *lower_term, const char *lower_doc, int state) {
    if (!lower_term || !lower_doc) return 0;
    int count = 0;
    const char *p = lower_doc;
    size_t tl = strlen(lower_term);
    if (tl == 0) return 0;
    while ((p = strstr(p, lower_term)) != NULL) {
        int is_start = (p == lower_doc || !isalnum((unsigned char)*(p - 1)));
        int is_end   = (*(p + tl) == '\0' || !isalnum((unsigned char)*(p + tl)));
        if (is_start && is_end) count++;
        p += tl;
    }
    return count;
}

// Approximate version used by func_substitute (kept defined; caller still calls exact)
int approx_tf_count_whole_word(const char *lower_term, const char *lower_doc, int state) {
    if (!lower_term || !lower_doc) return 0;
    // Cheap substring count without word-boundary checks
    int count = 0;
    const char *p = lower_doc;
    size_t tl = strlen(lower_term);
    if (tl == 0) return 0;
    while ((p = strstr(p, lower_term)) != NULL) {
        count++;
        p += tl;
    }
    return count;
}

char **read_documents_from_file(const char *filename, int *num_docs)
{
   FILE *file = fopen(filename, "r");
   if (!file)
   {
       fprintf(stderr, "Error: Cannot open file %s\n", filename);
       return NULL;
   }


   // First pass: count lines
   int line_count = 0;
   char buffer[4096];
   while (fgets(buffer, sizeof(buffer), file)  != NULL )
   {
       line_count++;
   }
   rewind(file);


   // Allocate memory for documents
   char **documents = (char **)malloc(line_count * sizeof(char *));
   if (!documents)
   {
       fclose(file);
       return NULL;
   }


   // Second pass: read documents
   int doc_index = 0;
   while (fgets(buffer, sizeof(buffer), file) != NULL && doc_index < line_count)
   {
       // Remove newline character
       size_t len = strlen(buffer);
       if (len > 0 && buffer[len - 1] == '\n')
       {
           buffer[len - 1] = '\0';
       }


       documents[doc_index] = strdup(buffer);
       if (!documents[doc_index])
       {
           // Clean up on error
           for (int i = 0; i < doc_index; i++)
               free(documents[i]);
           free(documents);
           fclose(file);
           return NULL;
       }
       doc_index++;
   }


   fclose(file);
   *num_docs = doc_index;
   return documents;
}


// ---------- KNOB 2: DF membership predicate (func_substitute) ----------
// Exact: check presence as a whole word (one hit is enough)
int df_contains_whole_word(const char *lower_term, const char *lower_doc, int state) {
    if (!lower_term || !lower_doc) return 0;
    const char *p = lower_doc;
    size_t tl = strlen(lower_term);
    if (tl == 0) return 0;
    while ((p = strstr(p, lower_term)) != NULL) {
        int is_start = (p == lower_doc || !isalnum((unsigned char)*(p - 1)));
        int is_end   = (*(p + tl) == '\0' || !isalnum((unsigned char)*(p + tl)));
        if (is_start && is_end) return 1;
        p += tl;
    }
    return 0;
}

// Approx: allow any substring match
int approx_df_contains_whole_word(const char *lower_term, const char *lower_doc, int state) {
    if (!lower_term || !lower_doc) return 0;
    return strstr(lower_doc, lower_term) ? 1 : 0;
}

// ---------- DF over corpus (uses KNOB 2 internally) ----------
int calculate_df(const char *term, char **corpus, int num_docs) {
    if (term == NULL || corpus == NULL) return 0;

    // lowercase term once
    char *lt = lower_dup(term);
    if (!lt) return 0;

    int count = 0;
    // choose a caller-side state ~= term length (bounded)
    int state = (int)strlen(lt);
    if (state > 16) state = 16;

    for (int i = 0; i < num_docs; ++i) {
        char *doc_copy = strdup(corpus[i]);
        if (!doc_copy) continue;
        for (char *p = doc_copy; *p; ++p) *p = (char)tolower((unsigned char)*p);

        // call exact DF membership (knobbed via func_substitute)
        if (df_contains_whole_word(lt, doc_copy, state)) count++;

        free(doc_copy);
    }
    free(lt);
    return count;
}

// ---------- KNOB 3: per-term scoring loop (loop_perforate) ----------
// Adds the BM25 contribution of 'lower_term' to each document's score.
void score_term_over_docs(
    const char *lower_term,
    char **lower_corpus,
    const double *doc_lengths,
    double avg_doc_len,
    double idf,
    DocumentScore *scores,
    int num_docs,
    int state // caller passes something like num_docs
){
    // The loop body is exact; loop_perforate will adjust the step in MLIR.
    for (int i = 0; i < num_docs; ++i) {
        int tf_state = (int)strlen(lower_term);
        if (tf_state > 16) tf_state = 16;

        int tf = tf_count_whole_word(lower_term, lower_corpus[i], tf_state);

        double numerator   = (double)tf * (K1 + 1.0);
        double denominator = (double)tf + K1 * (1.0 - B + B * (doc_lengths[i] / avg_doc_len));
        double term_score  = idf * (denominator > 0.0 ? (numerator / denominator) : 0.0);

        scores[i].score += term_score;
    }
}

// ---------- Main BM25 Ranking (refactored, no knobs here) ----------
DocumentScore* rank_documents_bm25(char *query, char **corpus, int num_docs) {
    if (query == NULL || corpus == NULL || num_docs <= 0) return NULL;

    // 1) preprocess: lengths + lowercased docs
    double *doc_lengths = (double *)malloc((size_t)num_docs * sizeof(double));
    char  **lower_corpus = (char  **)malloc((size_t)num_docs * sizeof(char*));
    if (!doc_lengths || !lower_corpus) {
        free(doc_lengths); free(lower_corpus);
        return NULL;
    }

    double total_len = 0.0;
    for (int i = 0; i < num_docs; ++i) {
        char *doc_copy = strdup(corpus[i]);
        if (!doc_copy) {
            for (int j = 0; j < i; ++j) free(lower_corpus[j]);
            free(lower_corpus); free(doc_lengths);
            return NULL;
        }
        doc_lengths[i] = (double)count_and_lower_words(doc_copy);
        total_len     += doc_lengths[i];
        lower_corpus[i] = doc_copy;
    }
    double avg_doc_len = total_len / (double)num_docs;

    // 2) scores
    DocumentScore *scores = (DocumentScore *)malloc((size_t)num_docs * sizeof(DocumentScore));
    if (!scores) {
        for (int i = 0; i < num_docs; ++i) free(lower_corpus[i]);
        free(lower_corpus); free(doc_lengths);
        return NULL;
    }
    for (int i = 0; i < num_docs; ++i) { scores[i].doc_index = i; scores[i].score = 0.0; }

    // 3) tokenize query (track uniques)
    char *query_copy = strdup(query);
    if (!query_copy) {
        for (int i = 0; i < num_docs; ++i) free(lower_corpus[i]);
        free(lower_corpus); free(doc_lengths); free(scores);
        return NULL;
    }

    char *processed_terms[256];
    int    processed_count = 0;

    char *rest = query_copy;
    const char *delims = " .,;:!?\"\'\n\t()[]{}<>";

    while (1) {
        char *term = strtok_r(rest, delims, &rest);
        if (!term) break;
        if (*term == '\0') continue;

        // de-dup (case-insensitive)
        int seen = 0;
        for (int k = 0; k < processed_count; ++k) {
            if (compare_tokens(term, processed_terms[k]) == 0) { seen = 1; break; }
        }
        if (seen) continue;

        if (processed_count >= 256) break;
        processed_terms[processed_count] = strdup(term);
        if (!processed_terms[processed_count]) continue;
        processed_count++;

        // compute DF and IDF (DF uses KNOB 2 within)
        int df = calculate_df(term, corpus, num_docs);
        if (df == 0) continue;
        double idf = calculate_idf(df, num_docs);

        // lowercase term once, then score across docs via KNOB 3
        char *lower_term = lower_dup(term);
        if (!lower_term) continue;

        int loop_state = num_docs; // caller-side state for loop_perforate
        score_term_over_docs(lower_term, lower_corpus, doc_lengths,
                             avg_doc_len, idf, scores, num_docs, loop_state);
        free(lower_term);
    }

    // cleanup temps
    free(query_copy);
    for (int i = 0; i < processed_count; ++i) free(processed_terms[i]);
    for (int i = 0; i < num_docs; ++i) free(lower_corpus[i]);
    free(lower_corpus);
    free(doc_lengths);

    // NOTE: we intentionally DO NOT sort, to match your current behavior.
    return scores;
}

void free_corpus(char **corpus, int num_docs) {
    if (!corpus) return;
    for (int i=0;i<num_docs;++i) free(corpus[i]);
    free(corpus);
}

// ---------- demo main ----------
static void usage(const char* prog) {
    fprintf(stderr, "Usage: %s <num_docs> [seed] [\"query string\"]\n", prog);
    fprintf(stderr, "  num_docs     : number of documents to generate (default 6)\n");
    fprintf(stderr, "  seed         : RNG seed (default 42)\n");
    fprintf(stderr, "  query string : BM25 query (default \"quick brown fox\")\n");
}

int main(int argc, char **argv)
{
   if (argc != 3)
   {
       fprintf(stderr, "Usage: %s <documents_file> <query>\n", argv[0]);
       fprintf(stderr, "  documents_file: File containing documents (one per line)\n");
       fprintf(stderr, "  query: BM25 query string\n");
       return 1;
   }


   char *documents_file = argv[1];
   char *query = argv[2];


   printf("Query: \"%s\"\n", query);
   printf("Reading documents from: %s\n\n", documents_file);


   // Read documents from file
   int num_docs;
   char **corpus = read_documents_from_file(documents_file, &num_docs);
   if (!corpus)
   {
       fprintf(stderr, "Failed to read documents from file.\n");
       return 1;
   }


   printf("Loaded %d documents\n", num_docs);


   // Rank documents using BM25
   struct timespec start, end;
   clock_gettime(CLOCK_MONOTONIC, &start);
   DocumentScore *ranked_scores = rank_documents_bm25(query, corpus, num_docs);
   clock_gettime(CLOCK_MONOTONIC, &end);
   double elapsed_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                       (end.tv_nsec - start.tv_nsec) / 1.0e6;


   if (ranked_scores)
   {
       printf("Ranking results:\n");
       for (int i = 0; i < num_docs; ++i)
       {
           int doc_index = ranked_scores[i].doc_index;
           printf("Rank %d: Doc %d (Score: %.4f) - \"%s\"\n",
                  i + 1,
                  doc_index,
                  ranked_scores[i].score,
                  corpus[doc_index]);
       }
       free(ranked_scores);
   }
   else
   {
       printf("An error occurred during ranking.\n");
   }

   printf("\nComputation time: %.3f ms\n", elapsed_ms);

   return 0;
}
