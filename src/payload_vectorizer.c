
#include "payload_vectorizer.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

/* =========================================================
 * Config interna
 * ========================================================= */

#define MAX_KNOWN_MERCHANTS 256
#define MAX_STR 128

/* =========================================================
 * Helpers gerais
 * ========================================================= */

static float clamp01(float x) {
    if (x < 0.0) return 0.0;
    if (x > 1.0) return 1.0;
    return x;
}

static char *read_text_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }

    long sz = ftell(f);
    if (sz < 0) {
        fclose(f);
        return NULL;
    }

    rewind(f);

    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }

    size_t nread = fread(buf, 1, (size_t)sz, f);
    if (nread != (size_t)sz) {
        free(buf);
        fclose(f);
        return NULL;
    }

    buf[sz] = '\0';
    fclose(f);
    return buf;
}

/* =========================================================
 * Cursor JSON
 * ========================================================= */

static void skip_ws_p(const char **p) {
    while (**p && isspace((unsigned char)**p)) {
        (*p)++;
    }
}

static int consume_char_p(const char **p, char expected) {
    skip_ws_p(p);
    if (**p != expected) {
        return -1;
    }
    (*p)++;
    return 0;
}

static int parse_json_string(
    const char **p,
    char *out,
    size_t out_size
) {
    skip_ws_p(p);

    if (**p != '"') {
        return -1;
    }

    (*p)++;

    size_t i = 0;

    while (**p) {
        unsigned char c = (unsigned char)**p;

        if (c == '"') {
            (*p)++;
            if (out) {
                if (out_size == 0) return -1;
                out[i] = '\0';
            }
            return 0;
        }

        if (c == '\\') {
            return -1; /* formato do desafio não precisa de escapes */
        }

        if (c < 0x20) {
            return -1;
        }

        if (out) {
            if (i + 1 >= out_size) {
                return -1;
            }
            out[i++] = (char)c;
        }

        (*p)++;
    }

    return -1;
}

static int parse_json_number(
    const char **p,
    float *out
) {
    skip_ws_p(p);

    errno = 0;
    char *end = NULL;
    float v = strtod(*p, &end);

    if (end == *p || errno != 0 || !isfinite(v)) {
        return -1;
    }

    *out = v;
    *p = end;
    return 0;
}

static int parse_json_int(
    const char **p,
    int *out
) {
    skip_ws_p(p);

    errno = 0;
    char *end = NULL;
    long v = strtol(*p, &end, 10);

    if (end == *p || errno != 0 || v < INT_MIN || v > INT_MAX) {
        return -1;
    }

    *out = (int)v;
    *p = end;
    return 0;
}

static int parse_json_bool(
    const char **p,
    int *out
) {
    skip_ws_p(p);

    if (strncmp(*p, "true", 4) == 0) {
        *out = 1;
        *p += 4;
        return 0;
    }

    if (strncmp(*p, "false", 5) == 0) {
        *out = 0;
        *p += 5;
        return 0;
    }

    return -1;
}

static int parse_json_null(const char **p) {
    skip_ws_p(p);

    if (strncmp(*p, "null", 4) != 0) {
        return -1;
    }

    *p += 4;
    return 0;
}

static int skip_json_value(const char **p);

static int skip_json_array(const char **p) {
    if (consume_char_p(p, '[') < 0) return -1;

    skip_ws_p(p);
    if (**p == ']') {
        (*p)++;
        return 0;
    }

    while (1) {
        if (skip_json_value(p) < 0) return -1;

        skip_ws_p(p);
        if (**p == ',') {
            (*p)++;
            continue;
        }
        if (**p == ']') {
            (*p)++;
            return 0;
        }
        return -1;
    }
}

static int skip_json_object(const char **p) {
    if (consume_char_p(p, '{') < 0) return -1;

    skip_ws_p(p);
    if (**p == '}') {
        (*p)++;
        return 0;
    }

    while (1) {
        char key[MAX_STR];

        if (parse_json_string(p, key, sizeof(key)) < 0) return -1;
        if (consume_char_p(p, ':') < 0) return -1;
        if (skip_json_value(p) < 0) return -1;

        skip_ws_p(p);
        if (**p == ',') {
            (*p)++;
            continue;
        }
        if (**p == '}') {
            (*p)++;
            return 0;
        }
        return -1;
    }
}

static int skip_json_value(const char **p) {
    skip_ws_p(p);

    if (**p == '"') {
        return parse_json_string(p, NULL, 0);
    }

    if (**p == '{') {
        return skip_json_object(p);
    }

    if (**p == '[') {
        return skip_json_array(p);
    }

    if (strncmp(*p, "true", 4) == 0 || strncmp(*p, "false", 5) == 0) {
        int dummy;
        return parse_json_bool(p, &dummy);
    }

    if (strncmp(*p, "null", 4) == 0) {
        return parse_json_null(p);
    }

    {
        float dummy;
        return parse_json_number(p, &dummy);
    }
}

/* =========================================================
 * Timestamp ISO 8601 UTC
 * formato: YYYY-MM-DDTHH:MM:SSZ
 * ========================================================= */

typedef struct {
    int year;
    int month;
    int day;
    int hour;
    int minute;
    int second;
} DateTimeUTC;

static int is_leap_year(int y) {
    return ((y % 4 == 0) && (y % 100 != 0)) || (y % 400 == 0);
}

static int days_in_month(int y, int m) {
    static const int dim[] = {
        31, 28, 31, 30, 31, 30,
        31, 31, 30, 31, 30, 31
    };

    if (m < 1 || m > 12) return 0;
    if (m == 2 && is_leap_year(y)) return 29;
    return dim[m - 1];
}

static int parse_iso8601_utc(
    const char *s,
    DateTimeUTC *dt
) {
    if (!s || !dt) return -1;

    if (strlen(s) != 20) {
        return -1;
    }

    /* YYYY-MM-DDTHH:MM:SSZ */
    for (int i = 0; i < 20; i++) {
        if (i == 4 || i == 7) {
            if (s[i] != '-') return -1;
        } else if (i == 10) {
            if (s[i] != 'T') return -1;
        } else if (i == 13 || i == 16) {
            if (s[i] != ':') return -1;
        } else if (i == 19) {
            if (s[i] != 'Z') return -1;
        } else {
            if (!isdigit((unsigned char)s[i])) return -1;
        }
    }

    int year =
        (s[0] - '0') * 1000 +
        (s[1] - '0') * 100 +
        (s[2] - '0') * 10 +
        (s[3] - '0');

    int month =
        (s[5] - '0') * 10 +
        (s[6] - '0');

    int day =
        (s[8] - '0') * 10 +
        (s[9] - '0');

    int hour =
        (s[11] - '0') * 10 +
        (s[12] - '0');

    int minute =
        (s[14] - '0') * 10 +
        (s[15] - '0');

    int second =
        (s[17] - '0') * 10 +
        (s[18] - '0');

    if (month < 1 || month > 12) return -1;
    if (day < 1 || day > days_in_month(year, month)) return -1;
    if (hour < 0 || hour > 23) return -1;
    if (minute < 0 || minute > 59) return -1;
    if (second < 0 || second > 59) return -1;

    dt->year = year;
    dt->month = month;
    dt->day = day;
    dt->hour = hour;
    dt->minute = minute;
    dt->second = second;

    return 0;
}

static int days_from_civil(int y, unsigned m, unsigned d) {
    y -= (m <= 2);
    const int era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = (unsigned)(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + (int)doe - 719468;
}

/* segunda=0 ... domingo=6 */
static int weekday_monday0(int year, int month, int day) {
    int days = days_from_civil(year, (unsigned)month, (unsigned)day);
    int w = (days + 3) % 7;
    if (w < 0) w += 7;
    return w;
}

static long long datetime_to_seconds(const DateTimeUTC *dt) {
    int days = days_from_civil(dt->year, (unsigned)dt->month, (unsigned)dt->day);
    long long s = (long long)days * 86400LL;
    s += (long long)dt->hour * 3600LL;
    s += (long long)dt->minute * 60LL;
    s += (long long)dt->second;
    return s;
}

/* =========================================================
 * Estrutura interna do payload
 * ========================================================= */

typedef struct {
    int have_transaction;
    int have_customer;
    int have_merchant;
    int have_terminal;
    int have_last_transaction;

    float amount;
    int installments;
    DateTimeUTC requested_at;
    int have_requested_at;

    float avg_amount;
    int tx_count_24h;
    int have_avg_amount;
    int have_tx_count_24h;

    char merchant_id[MAX_STR];
    char merchant_mcc[16];
    float merchant_avg_amount;
    int have_merchant_id;
    int have_merchant_mcc;
    int have_merchant_avg_amount;

    float km_from_home;
    int is_online;
    int card_present;
    int have_km_from_home;
    int have_is_online;
    int have_card_present;

    char known_merchants[MAX_KNOWN_MERCHANTS][MAX_STR];
    int known_count;

    int last_is_null;
    int have_last_requested_at;
    int have_last_minutes_since_last_tx;
    int have_last_km_from_tx;
    DateTimeUTC last_requested_at;
    float minutes_since_last_tx;
    float km_from_last_tx;
} PayloadState;

/* =========================================================
 * known_merchants: array de strings
 * ========================================================= */

static int parse_known_merchants_array(
    const char **p,
    PayloadState *st
) {
    if (consume_char_p(p, '[') < 0) {
        return -1;
    }

    skip_ws_p(p);
    if (**p == ']') {
        (*p)++;
        return 0;
    }

    while (1) {
        if (st->known_count >= MAX_KNOWN_MERCHANTS) {
            return -1;
        }

        if (parse_json_string(
                p,
                st->known_merchants[st->known_count],
                sizeof(st->known_merchants[0])
            ) < 0) {
            return -1;
        }

        st->known_count++;

        skip_ws_p(p);
        if (**p == ',') {
            (*p)++;
            continue;
        }

        if (**p == ']') {
            (*p)++;
            return 0;
        }

        return -1;
    }
}

static int merchant_is_known(
    const PayloadState *st,
    const char *merchant_id
) {
    for (int i = 0; i < st->known_count; i++) {
        if (strcmp(st->known_merchants[i], merchant_id) == 0) {
            return 1;
        }
    }
    return 0;
}

/* =========================================================
 * Parse sections
 * ========================================================= */

static int parse_transaction_object(
    const char **p,
    PayloadState *st
) {
    if (consume_char_p(p, '{') < 0) return -1;

    while (1) {
        skip_ws_p(p);
        if (**p == '}') {
            (*p)++;
            st->have_transaction = 1;
            return 0;
        }

        char key[MAX_STR];
        if (parse_json_string(p, key, sizeof(key)) < 0) return -1;
        if (consume_char_p(p, ':') < 0) return -1;

        if (strcmp(key, "amount") == 0) {
            if (parse_json_number(p, &st->amount) < 0) return -1;
        }
        else if (strcmp(key, "installments") == 0) {
            if (parse_json_int(p, &st->installments) < 0) return -1;
        }
        else if (strcmp(key, "requested_at") == 0) {
            char ts[MAX_STR];
            if (parse_json_string(p, ts, sizeof(ts)) < 0) return -1;
            if (parse_iso8601_utc(ts, &st->requested_at) < 0) return -1;
            st->have_requested_at = 1;
        }
        else {
            if (skip_json_value(p) < 0) return -1;
        }

        skip_ws_p(p);
        if (**p == ',') {
            (*p)++;
            continue;
        }
        if (**p == '}') {
            (*p)++;
            st->have_transaction = 1;
            return 0;
        }
        return -1;
    }
}

static int parse_customer_object(
    const char **p,
    PayloadState *st
) {
    if (consume_char_p(p, '{') < 0) return -1;

    while (1) {
        skip_ws_p(p);
        if (**p == '}') {
            (*p)++;
            st->have_customer = 1;
            return 0;
        }

        char key[MAX_STR];
        if (parse_json_string(p, key, sizeof(key)) < 0) return -1;
        if (consume_char_p(p, ':') < 0) return -1;

        if (strcmp(key, "avg_amount") == 0) {
            if (parse_json_number(p, &st->avg_amount) < 0) return -1;
            st->have_avg_amount = 1;
        }
        else if (strcmp(key, "tx_count_24h") == 0) {
            if (parse_json_int(p, &st->tx_count_24h) < 0) return -1;
            st->have_tx_count_24h = 1;
        }
        else if (strcmp(key, "known_merchants") == 0) {
            if (parse_known_merchants_array(p, st) < 0) return -1;
        }
        else {
            if (skip_json_value(p) < 0) return -1;
        }

        skip_ws_p(p);
        if (**p == ',') {
            (*p)++;
            continue;
        }
        if (**p == '}') {
            (*p)++;
            st->have_customer = 1;
            return 0;
        }
        return -1;
    }
}

static int parse_merchant_object(
    const char **p,
    PayloadState *st
) {
    if (consume_char_p(p, '{') < 0) return -1;

    while (1) {
        skip_ws_p(p);
        if (**p == '}') {
            (*p)++;
            st->have_merchant = 1;
            return 0;
        }

        char key[MAX_STR];
        if (parse_json_string(p, key, sizeof(key)) < 0) return -1;
        if (consume_char_p(p, ':') < 0) return -1;

        if (strcmp(key, "id") == 0) {
            if (parse_json_string(p, st->merchant_id, sizeof(st->merchant_id)) < 0) return -1;
            st->have_merchant_id = 1;
        }
        else if (strcmp(key, "mcc") == 0) {
            if (parse_json_string(p, st->merchant_mcc, sizeof(st->merchant_mcc)) < 0) return -1;
            st->have_merchant_mcc = 1;
        }
        else if (strcmp(key, "avg_amount") == 0) {
            if (parse_json_number(p, &st->merchant_avg_amount) < 0) return -1;
            st->have_merchant_avg_amount = 1;
        }
        else {
            if (skip_json_value(p) < 0) return -1;
        }

        skip_ws_p(p);
        if (**p == ',') {
            (*p)++;
            continue;
        }
        if (**p == '}') {
            (*p)++;
            st->have_merchant = 1;
            return 0;
        }
        return -1;
    }
}

static int parse_terminal_object(
    const char **p,
    PayloadState *st
) {
    if (consume_char_p(p, '{') < 0) return -1;

    while (1) {
        skip_ws_p(p);
        if (**p == '}') {
            (*p)++;
            st->have_terminal = 1;
            return 0;
        }

        char key[MAX_STR];
        if (parse_json_string(p, key, sizeof(key)) < 0) return -1;
        if (consume_char_p(p, ':') < 0) return -1;

        if (strcmp(key, "km_from_home") == 0) {
            if (parse_json_number(p, &st->km_from_home) < 0) return -1;
            st->have_km_from_home = 1;
        }
        else if (strcmp(key, "is_online") == 0) {
            if (parse_json_bool(p, &st->is_online) < 0) return -1;
            st->have_is_online = 1;
        }
        else if (strcmp(key, "card_present") == 0) {
            if (parse_json_bool(p, &st->card_present) < 0) return -1;
            st->have_card_present = 1;
        }
        else {
            if (skip_json_value(p) < 0) return -1;
        }

        skip_ws_p(p);
        if (**p == ',') {
            (*p)++;
            continue;
        }
        if (**p == '}') {
            (*p)++;
            st->have_terminal = 1;
            return 0;
        }
        return -1;
    }
}

static int parse_last_transaction_value(
    const char **p,
    PayloadState *st
) {
    skip_ws_p(p);

    if (strncmp(*p, "null", 4) == 0) {
        *p += 4;
        st->have_last_transaction = 1;
        st->last_is_null = 1;
        return 0;
    }

    if (consume_char_p(p, '{') < 0) {
        return -1;
    }

    st->have_last_transaction = 1;
    st->last_is_null = 0;

    while (1) {
        skip_ws_p(p);
        if (**p == '}') {
            (*p)++;
            return 0;
        }

        char key[MAX_STR];
        if (parse_json_string(p, key, sizeof(key)) < 0) return -1;
        if (consume_char_p(p, ':') < 0) return -1;

        if (strcmp(key, "requested_at") == 0 ||
    strcmp(key, "timestamp") == 0) {
            char ts[MAX_STR];
            if (parse_json_string(p, ts, sizeof(ts)) < 0) return -1;
            if (parse_iso8601_utc(ts, &st->last_requested_at) < 0) return -1;
            st->have_last_requested_at = 1;
        }
        else if (strcmp(key, "minutes_since_last_tx") == 0) {
            if (parse_json_number(p, &st->minutes_since_last_tx) < 0) return -1;
            st->have_last_minutes_since_last_tx = 1;
        }
        else if (strcmp(key, "km_from_current") == 0 ||
                 strcmp(key, "km_from_last_tx") == 0) {
            if (parse_json_number(p, &st->km_from_last_tx) < 0) return -1;
            st->have_last_km_from_tx = 1;
        }
        else {
            if (skip_json_value(p) < 0) return -1;
        }

        skip_ws_p(p);
        if (**p == ',') {
            (*p)++;
            continue;
        }
        if (**p == '}') {
            (*p)++;
            return 0;
        }
        return -1;
    }
}

static int parse_payload(
    const char *json,
    PayloadState *st
) {
    const char *p = json;

    if (consume_char_p(&p, '{') < 0) return -1;

    while (1) {
        skip_ws_p(&p);
        if (*p == '}') {
            p++;
            break;
        }

        char key[MAX_STR];
        if (parse_json_string(&p, key, sizeof(key)) < 0) return -1;
        if (consume_char_p(&p, ':') < 0) return -1;

        if (strcmp(key, "transaction") == 0) {
            if (st->have_transaction) return -1;
            if (parse_transaction_object(&p, st) < 0) return -1;
        }
        else if (strcmp(key, "customer") == 0) {
            if (st->have_customer) return -1;
            if (parse_customer_object(&p, st) < 0) return -1;
        }
        else if (strcmp(key, "merchant") == 0) {
            if (st->have_merchant) return -1;
            if (parse_merchant_object(&p, st) < 0) return -1;
        }
        else if (strcmp(key, "terminal") == 0) {
            if (st->have_terminal) return -1;
            if (parse_terminal_object(&p, st) < 0) return -1;
        }
        else if (strcmp(key, "last_transaction") == 0) {
            if (st->have_last_transaction) return -1;
            if (parse_last_transaction_value(&p, st) < 0) return -1;
        }
        else {
            if (skip_json_value(&p) < 0) return -1;
        }

        skip_ws_p(&p);
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

    if (!st->have_transaction ||
        !st->have_customer ||
        !st->have_merchant ||
        !st->have_terminal ||
        !st->have_last_transaction) {
        return -1;
    }

    if (!st->have_requested_at ||
        !st->have_avg_amount ||
        !st->have_tx_count_24h ||
        !st->have_merchant_id ||
        !st->have_merchant_mcc ||
        !st->have_merchant_avg_amount ||
        !st->have_km_from_home ||
        !st->have_is_online ||
        !st->have_card_present) {
        return -1;
    }

    if (st->avg_amount <= 0.0 || st->merchant_avg_amount < 0.0) {
        return -1;
    }

    return 0;
}

/* =========================================================
 * MCC risk table
 * ========================================================= */

static int parse_mcc_risk_json(
    const char *json,
    MccRiskTable *table
) {
    const char *p = json;
    size_t cap = 16;

    table->entries = (MccRiskEntry *)malloc(sizeof(MccRiskEntry) * cap);
    if (!table->entries) {
        return -1;
    }

    table->count = 0;

    if (consume_char_p(&p, '{') < 0) {
        free(table->entries);
        table->entries = NULL;
        table->count = 0;
        return -1;
    }

    skip_ws_p(&p);
    if (*p == '}') {
        p++;
        free(table->entries);
        table->entries = NULL;
        table->count = 0;
        return -1;
    }

    while (1) {
        char key[16];
        if (parse_json_string(&p, key, sizeof(key)) < 0) {
            free(table->entries);
            table->entries = NULL;
            table->count = 0;
            return -1;
        }

        if (strlen(key) != 4) {
            free(table->entries);
            table->entries = NULL;
            table->count = 0;
            return -1;
        }

        for (int i = 0; i < 4; i++) {
            if (!isdigit((unsigned char)key[i])) {
                free(table->entries);
                table->entries = NULL;
                table->count = 0;
                return -1;
            }
        }

        if (consume_char_p(&p, ':') < 0) {
            free(table->entries);
            table->entries = NULL;
            table->count = 0;
            return -1;
        }

        float risk = 0.0;
        if (parse_json_number(&p, &risk) < 0 || risk < 0.0 || risk > 1.0) {
            free(table->entries);
            table->entries = NULL;
            table->count = 0;
            return -1;
        }

        if (table->count >= (int)cap) {
            cap *= 2;
            MccRiskEntry *new_entries = (MccRiskEntry *)realloc(table->entries, sizeof(MccRiskEntry) * cap);
            if (!new_entries) {
                free(table->entries);
                table->entries = NULL;
                table->count = 0;
                return -1;
            }
            table->entries = new_entries;
        }

        snprintf(table->entries[table->count].mcc, sizeof(table->entries[table->count].mcc), "%s", key);
        table->entries[table->count].risk = risk;
        table->count++;

        skip_ws_p(&p);
        if (*p == ',') {
            p++;
            continue;
        }
        if (*p == '}') {
            p++;
            break;
        }

        free(table->entries);
        table->entries = NULL;
        table->count = 0;
        return -1;
    }

    return 0;
}

static float lookup_mcc(
    const MccRiskTable *table,
    const char *mcc
) {
    for (int i = 0; i < table->count; i++) {
        if (strcmp(table->entries[i].mcc, mcc) == 0) {
            return table->entries[i].risk;
        }
    }
    return 0.5;
}

/* =========================================================
 * Normalization JSON
 * ========================================================= */

static int parse_normalization_json(
    const char *json,
    NormalizationConfig *norm
) {
    const char *p = json;
    int seen_max_amount = 0;
    int seen_max_installments = 0;
    int seen_amount_vs_avg_ratio = 0;
    int seen_max_minutes = 0;
    int seen_max_km = 0;
    int seen_max_tx_count_24h = 0;
    int seen_max_merchant_avg_amount = 0;

    if (consume_char_p(&p, '{') < 0) return -1;

    while (1) {
        skip_ws_p(&p);
        if (*p == '}') {
            p++;
            break;
        }

        char key[MAX_STR];
        if (parse_json_string(&p, key, sizeof(key)) < 0) return -1;
        if (consume_char_p(&p, ':') < 0) return -1;

        float v = 0.0;
        if (parse_json_number(&p, &v) < 0) return -1;

        if (strcmp(key, "max_amount") == 0) {
            norm->max_amount = v;
            seen_max_amount = 1;
        }
        else if (strcmp(key, "max_installments") == 0) {
            norm->max_installments = v;
            seen_max_installments = 1;
        }
        else if (strcmp(key, "amount_vs_avg_ratio") == 0) {
            norm->amount_vs_avg_ratio = v;
            seen_amount_vs_avg_ratio = 1;
        }
        else if (strcmp(key, "max_minutes") == 0) {
            norm->max_minutes = v;
            seen_max_minutes = 1;
        }
        else if (strcmp(key, "max_km") == 0) {
            norm->max_km = v;
            seen_max_km = 1;
        }
        else if (strcmp(key, "max_tx_count_24h") == 0) {
            norm->max_tx_count_24h = v;
            seen_max_tx_count_24h = 1;
        }
        else if (strcmp(key, "max_merchant_avg_amount") == 0) {
            norm->max_merchant_avg_amount = v;
            seen_max_merchant_avg_amount = 1;
        }
        else {
            return -1;
        }

        skip_ws_p(&p);
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

    if (!seen_max_amount ||
        !seen_max_installments ||
        !seen_amount_vs_avg_ratio ||
        !seen_max_minutes ||
        !seen_max_km ||
        !seen_max_tx_count_24h ||
        !seen_max_merchant_avg_amount) {
        return -1;
    }

    if (norm->max_amount <= 0.0 ||
        norm->max_installments <= 0.0 ||
        norm->amount_vs_avg_ratio <= 0.0 ||
        norm->max_minutes <= 0.0 ||
        norm->max_km <= 0.0 ||
        norm->max_tx_count_24h <= 0.0 ||
        norm->max_merchant_avg_amount <= 0.0) {
        return -1;
    }

    return 0;
}

/* =========================================================
 * Init / Destroy
 * ========================================================= */

int vectorizer_init(
    VectorizerContext *ctx,
    const char *normalization_path,
    const char *mcc_risk_path
) {
    if (!ctx || !normalization_path || !mcc_risk_path) {
        return -1;
    }

    memset(ctx, 0, sizeof(*ctx));

    char *norm_json = read_text_file(normalization_path);
    if (!norm_json) {
        return -1;
    }

    if (parse_normalization_json(norm_json, &ctx->norm) < 0) {
        free(norm_json);
        return -1;
    }

    free(norm_json);

    char *mcc_json = read_text_file(mcc_risk_path);
    if (!mcc_json) {
        vectorizer_destroy(ctx);
        return -1;
    }

    if (parse_mcc_risk_json(mcc_json, &ctx->mcc_table) < 0) {
        free(mcc_json);
        vectorizer_destroy(ctx);
        return -1;
    }

    free(mcc_json);
    return 0;
}

void vectorizer_destroy(VectorizerContext *ctx) {
    if (!ctx) return;

    free(ctx->mcc_table.entries);
    ctx->mcc_table.entries = NULL;
    ctx->mcc_table.count = 0;
}

/* =========================================================
 * Build vector
 * ========================================================= */

int build_transaction_vector(
    VectorizerContext *ctx,
    const char *json_payload,
    float out_vector[VECTOR_SIZE]
) {
    if (!ctx || !json_payload || !out_vector || !ctx->mcc_table.entries) {
        return -1;
    }

    PayloadState st;
    memset(&st, 0, sizeof(st));

    if (parse_payload(json_payload, &st) < 0) {
        //fprintf(stderr, "aqui payload");
        return -1;
    }

    /* minutes_since_last_tx / km_from_last_tx */
    if (st.last_is_null) {
        out_vector[5] = -1.0;
        out_vector[6] = -1.0;
    } else {
        if (!st.have_last_km_from_tx) {
            //fprintf(stderr, "aqui payload 2");
            return -1;
        }

        if (!st.have_last_minutes_since_last_tx) {
            if (!st.have_last_requested_at || !st.have_requested_at) {
                //fprintf(stderr, "aqui payload 3");
                return -1;
            }

            long long current_s = datetime_to_seconds(&st.requested_at);
            long long last_s = datetime_to_seconds(&st.last_requested_at);

            if (current_s < last_s) {
                //fprintf(stderr, "aqui payload");
                return -1;
            }

            st.minutes_since_last_tx = (float)(current_s - last_s) / 60.0;
            st.have_last_minutes_since_last_tx = 1;
        }

        out_vector[5] = (float)clamp01(
            st.minutes_since_last_tx / ctx->norm.max_minutes
        );

        out_vector[6] = (float)clamp01(
            st.km_from_last_tx / ctx->norm.max_km
        );
    }

    /* 0..4 */
    out_vector[0] = (float)clamp01(st.amount / ctx->norm.max_amount);
    out_vector[1] = (float)clamp01((float)st.installments / ctx->norm.max_installments);
    out_vector[2] = (float)clamp01((st.amount / st.avg_amount) / ctx->norm.amount_vs_avg_ratio);
    out_vector[3] = (float)(st.requested_at.hour / 23.0);
    out_vector[4] = (float)(weekday_monday0(st.requested_at.year, st.requested_at.month, st.requested_at.day) / 6.0);

    /* 7..13 */
    out_vector[7] = (float)clamp01(st.km_from_home / ctx->norm.max_km);
    out_vector[8] = (float)clamp01((float)st.tx_count_24h / ctx->norm.max_tx_count_24h);
    out_vector[9] = st.is_online ? 1.0 : 0.0;
    out_vector[10] = st.card_present ? 1.0 : 0.0;
    out_vector[11] = merchant_is_known(&st, st.merchant_id) ? 0.0 : 1.0;
    out_vector[12] = lookup_mcc(&ctx->mcc_table, st.merchant_mcc);
    out_vector[13] = (float)clamp01(st.merchant_avg_amount / ctx->norm.max_merchant_avg_amount);

    /* DEBUG */
    //fprintf(stderr,"VECTOR = [");

   

    //fprintf(stderr,"]\n");

    return 0;
}

