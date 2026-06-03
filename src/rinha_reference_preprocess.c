
/*
 * rinha_reference_preprocess.c
 *
 * Fluxo:
 *   1) resources/references.json.gz -> resources/example-references.json
 *   2) resources/example-references.json -> index.bin
 *
 * Uso:
 *   ./preprocess resources/references.json.gz index.bin
 *
 * Build:
 *   gcc -O3 -march=native -std=c99 -Wall -Wextra \
 *       rinha_reference_preprocess.c -lz -o preprocess
 */

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <zlib.h>

#define VECTOR_DIMS 14
#define CHUNK_SIZE 32768
#define OBJ_INIT_CAP 1024

#define JSON_OUT_PATH "resources/example-references.json"

/* ========================= utils ========================= */

static void die(const char *msg) {
    fprintf(stderr, "Erro: %s\n", msg);
    exit(EXIT_FAILURE);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) die("malloc falhou");
    return p;
}

static void *xrealloc(void *ptr, size_t n) {
    void *p = realloc(ptr, n);
    if (!p) die("realloc falhou");
    return p;
}

static const char *skip_ws(const char *p) {
    while (*p && isspace((unsigned char)*p))
        p++;
    return p;
}

static int ensure_resources_dir(void) {
    if (mkdir("resources", 0755) != 0 && errno != EEXIST) {
        perror("mkdir(resources)");
        return -1;
    }
    return 0;
}

/* ========================= formato binario ========================= */

#pragma pack(push, 1)

typedef struct {
    float vector[VECTOR_DIMS];
    uint8_t label; /* 0 = legit, 1 = fraud */
} BinRecord;

typedef struct {
    char magic[8];      /* "RNHBIN" + \0\0 */
    uint32_t version;
    uint32_t dims;
    uint64_t count;
} BinHeader;

#pragma pack(pop)

static void write_header_placeholder(FILE *f) {
    BinHeader h;
    memset(&h, 0, sizeof(h));

    memcpy(h.magic, "RNHBIN", 6);
    h.version = 1;
    h.dims = VECTOR_DIMS;
    h.count = 0;

    if (fwrite(&h, sizeof(h), 1, f) != 1) {
        die("falha ao escrever header placeholder");
    }
}

static void patch_header(FILE *f, uint64_t count) {
    BinHeader h;
    memset(&h, 0, sizeof(h));

    memcpy(h.magic, "RNHBIN", 6);
    h.version = 1;
    h.dims = VECTOR_DIMS;
    h.count = count;

    if (fseek(f, 0, SEEK_SET) != 0) {
        die("fseek falhou ao patchar header");
    }

    if (fwrite(&h, sizeof(h), 1, f) != 1) {
        die("falha ao patchar header");
    }
}

/* ========================= decompression ========================= */

static int decompress_gz_to_file(
    const char *gz_path,
    const char *json_out_path
) {
    gzFile gz = gzopen(gz_path, "rb");
    if (!gz) {
        fprintf(stderr, "Falha ao abrir %s\n", gz_path);
        return -1;
    }

    FILE *out = fopen(json_out_path, "wb");
    if (!out) {
        perror(json_out_path);
        gzclose(gz);
        return -1;
    }

    char chunk[CHUNK_SIZE];
    int rd;

    while ((rd = gzread(gz, chunk, sizeof(chunk))) > 0) {
        if (fwrite(chunk, 1, (size_t)rd, out) != (size_t)rd) {
            perror("fwrite");
            fclose(out);
            gzclose(gz);
            return -1;
        }
    }

    if (rd < 0) {
        int errnum = 0;
        const char *msg = gzerror(gz, &errnum);
        fprintf(stderr, "gzread falhou: %s\n", msg ? msg : "erro");
        fclose(out);
        gzclose(gz);
        return -1;
    }

    if (fflush(out) != 0) {
        perror("fflush");
        fclose(out);
        gzclose(gz);
        return -1;
    }

    fclose(out);
    gzclose(gz);

    return 0;
}

/* ========================= JSON parsing ========================= */

static const char *parse_json_string_token(
    const char *p,
    char *out,
    size_t cap
) {
    if (*p != '"')
        return NULL;

    p++;

    size_t len = 0;

    while (*p) {
        unsigned char c = (unsigned char)*p++;

        if (c == '"') {
            if (out) {
                if (cap == 0 || len >= cap)
                    return NULL;
                out[len] = '\0';
            }
            return p;
        }

        if (c == '\\') {
            unsigned char e = (unsigned char)*p++;
            if (!e)
                return NULL;

            switch (e) {
                case '"':  c = '"';  break;
                case '\\': c = '\\'; break;
                case '/':  c = '/';  break;
                case 'b':  c = '\b'; break;
                case 'f':  c = '\f'; break;
                case 'n':  c = '\n'; break;
                case 'r':  c = '\r'; break;
                case 't':  c = '\t'; break;
                case 'u':
                    /* intencionalmente estrito: não aceita \u */
                    return NULL;
                default:
                    return NULL;
            }
        }

        if (out) {
            if (cap == 0 || len + 1 >= cap)
                return NULL;
            out[len++] = (char)c;
        } else {
            len++;
        }
    }

    return NULL;
}

static const char *skip_json_block(const char *p) {
    if (*p != '{' && *p != '[')
        return NULL;

    char stack[256];
    size_t top = 0;
    int in_string = 0;
    int escape = 0;

    stack[top++] = *p;
    p++;

    while (*p) {
        char c = *p++;

        if (in_string) {
            if (escape) {
                escape = 0;
            } else if (c == '\\') {
                escape = 1;
            } else if (c == '"') {
                in_string = 0;
            }
            continue;
        }

        if (c == '"') {
            in_string = 1;
            continue;
        }

        if (c == '{' || c == '[') {
            if (top >= sizeof(stack))
                return NULL;
            stack[top++] = c;
            continue;
        }

        if (c == '}' || c == ']') {
            if (top == 0)
                return NULL;

            char open = stack[top - 1];
            if ((c == '}' && open != '{') ||
                (c == ']' && open != '[')) {
                return NULL;
            }

            top--;

            if (top == 0)
                return p;
        }
    }

    return NULL;
}

static const char *skip_json_value(const char *p) {
    p = skip_ws(p);

    if (!*p)
        return NULL;

    if (*p == '"')
        return parse_json_string_token(p, NULL, 0);

    if (*p == '{' || *p == '[')
        return skip_json_block(p);

    while (*p) {
        unsigned char c = (unsigned char)*p;
        if (isspace(c) || c == ',' || c == '}' || c == ']')
            break;
        p++;
    }

    return p;
}

static const char *parse_float_array(
    const char *p,
    float *out,
    int n
) {
    p = skip_ws(p);

    if (*p != '[')
        return NULL;

    p++;

    for (int i = 0; i < n; i++) {
        p = skip_ws(p);

        errno = 0;
        char *end = NULL;
        double v = strtod(p, &end);

        if (end == p || errno != 0 || !isfinite(v))
            return NULL;

        out[i] = (float)v;
        p = end;

        p = skip_ws(p);

        if (i != n - 1) {
            if (*p != ',')
                return NULL;
            p++;
        }
    }

    p = skip_ws(p);

    if (*p != ']')
        return NULL;

    return p + 1;
}

static const char *parse_label_value(
    const char *p,
    uint8_t *label
) {
    p = skip_ws(p);

    if (*p == '"') {
        char buf[32];
        p = parse_json_string_token(p, buf, sizeof(buf));
        if (!p)
            return NULL;

        if (strcmp(buf, "legit") == 0) {
            *label = 0;
            return p;
        }

        if (strcmp(buf, "fraud") == 0) {
            *label = 1;
            return p;
        }

        return NULL;
    }

    /* tolera label numérico 0/1, por segurança */
    if (*p == '0') {
        *label = 0;
        return p + 1;
    }

    if (*p == '1') {
        *label = 1;
        return p + 1;
    }

    errno = 0;
    char *end = NULL;
    long v = strtol(p, &end, 10);

    if (end == p || errno != 0 || (v != 0 && v != 1))
        return NULL;

    *label = (uint8_t)v;
    return end;
}

static int parse_reference_object(
    const char *obj,
    BinRecord *rec
) {
    memset(rec, 0, sizeof(*rec));

    int seen_vector = 0;
    int seen_label = 0;

    const char *p = skip_ws(obj);

    if (*p != '{')
        return -1;

    p++;

    while (1) {
        p = skip_ws(p);

        if (*p == '}') {
            p++;
            break;
        }

        if (*p != '"')
            return -1;

        char key[64];

        p = parse_json_string_token(p, key, sizeof(key));
        if (!p)
            return -1;

        p = skip_ws(p);

        if (*p != ':')
            return -1;

        p++;
        p = skip_ws(p);

        if (strcmp(key, "vector") == 0) {
            if (seen_vector)
                return -1;

            p = parse_float_array(p, rec->vector, VECTOR_DIMS);
            if (!p)
                return -1;

            seen_vector = 1;
        }
        else if (strcmp(key, "label") == 0) {
            if (seen_label)
                return -1;

            p = parse_label_value(p, &rec->label);
            if (!p)
                return -1;

            seen_label = 1;
        }
        else {
            p = skip_json_value(p);
            if (!p)
                return -1;
        }

        p = skip_ws(p);

        if (*p == ',') {
            p++;
            continue;
        }

        if (*p == '}') {
            p++;
            break;
        }

        return -1;
    }

    if (!seen_vector || !seen_label)
        return -1;

    for (int i = 0; i < VECTOR_DIMS; i++) {
        if (!isfinite(rec->vector[i]))
            return -1;
    }

    return 0;
}

/* ========================= object buffer ========================= */

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} ObjBuf;

static void objbuf_init(ObjBuf *b) {
    b->cap = OBJ_INIT_CAP;
    b->len = 0;
    b->data = xmalloc(b->cap);
}

static void objbuf_reset(ObjBuf *b) {
    b->len = 0;
}

static void objbuf_push(ObjBuf *b, char c) {
    if (b->len + 1 >= b->cap) {
        b->cap *= 2;
        b->data = xrealloc(b->data, b->cap);
    }

    b->data[b->len++] = c;
}

static void objbuf_free(ObjBuf *b) {
    free(b->data);
}

/* ========================= parse do JSON descompactado ========================= */

static int process_json_file(
    const char *json_path,
    FILE *out,
    uint64_t *count_out
) {
    FILE *in = fopen(json_path, "rb");
    if (!in) {
        perror(json_path);
        return -1;
    }

    char chunk[CHUNK_SIZE];

    ObjBuf obj;
    objbuf_init(&obj);

    int capturing = 0;
    int in_string = 0;
    int escape = 0;
    int depth = 0;

    uint64_t count = 0;

    uint64_t line = 1;
    uint64_t col = 0;
    uint64_t obj_start_line = 1;
    uint64_t obj_start_col = 1;

    size_t rd;

    while ((rd = fread(chunk, 1, sizeof(chunk), in)) > 0) {
        for (size_t i = 0; i < rd; i++) {
            char c = chunk[i];

            col++;
            if (c == '\n') {
                line++;
                col = 0;
            }

            if (!capturing) {
                if (c == '{') {
                    capturing = 1;
                    in_string = 0;
                    escape = 0;
                    depth = 1;

                    obj_start_line = line;
                    obj_start_col = col;

                    objbuf_reset(&obj);
                    objbuf_push(&obj, c);
                }
                continue;
            }

            objbuf_push(&obj, c);

            if (in_string) {
                if (escape) {
                    escape = 0;
                }
                else if (c == '\\') {
                    escape = 1;
                }
                else if (c == '"') {
                    in_string = 0;
                }
                continue;
            }

            if (c == '"') {
                in_string = 1;
            }
            else if (c == '{') {
                depth++;
            }
            else if (c == '}') {
                depth--;

                if (depth == 0) {
                    obj.data[obj.len] = '\0';

                    BinRecord rec;

                    if (parse_reference_object(obj.data, &rec) != 0) {
                        fprintf(stderr,
                                "Falha ao parsear objeto #%llu (inicio em %llu:%llu)\n",
                                (unsigned long long)(count + 1),
                                (unsigned long long)obj_start_line,
                                (unsigned long long)obj_start_col);
                        fclose(in);
                        objbuf_free(&obj);
                        return -1;
                    }

                    if (fwrite(&rec, sizeof(rec), 1, out) != 1) {
                        perror("fwrite");
                        fclose(in);
                        objbuf_free(&obj);
                        return -1;
                    }

                    count++;
                    capturing = 0;
                }
            }
        }
    }

    if (ferror(in)) {
        perror("fread");
        fclose(in);
        objbuf_free(&obj);
        return -1;
    }

    if (capturing || depth != 0) {
        fprintf(stderr, "JSON terminou com objeto incompleto\n");
        fclose(in);
        objbuf_free(&obj);
        return -1;
    }

    fclose(in);
    objbuf_free(&obj);

    *count_out = count;
    return 0;
}

/* ========================= main ========================= */

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
                "Uso:\n"
                "  %s <references.json.gz> <index.bin>\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    const char *gz_path = argv[1];
    const char *bin_path = argv[2];

    if (ensure_resources_dir() != 0)
        return EXIT_FAILURE;

    if (decompress_gz_to_file(gz_path, JSON_OUT_PATH) != 0)
        return EXIT_FAILURE;

    FILE *out = fopen(bin_path, "wb+");
    if (!out) {
        perror(bin_path);
        return EXIT_FAILURE;
    }

    write_header_placeholder(out);

    uint64_t count = 0;

    if (process_json_file(JSON_OUT_PATH, out, &count) != 0) {
        fclose(out);
        return EXIT_FAILURE;
    }

    patch_header(out, count);

    if (fflush(out) != 0) {
        perror("fflush");
        fclose(out);
        return EXIT_FAILURE;
    }

    fclose(out);

    fprintf(stderr,
            "OK: %s -> %s (%llu registros)\n",
            gz_path,
            bin_path,
            (unsigned long long)count);

    return EXIT_SUCCESS;
}

