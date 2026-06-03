#ifndef PAYLOAD_VECTORIZER_H
#define PAYLOAD_VECTORIZER_H

#include <stdint.h>

#define VECTOR_SIZE 14

typedef struct {
    float max_amount;
    float max_installments;
    float amount_vs_avg_ratio;
    float max_minutes;
    float max_km;
    float max_tx_count_24h;
    float max_merchant_avg_amount;
} NormalizationConfig;

typedef struct {
    char mcc[5];
    float risk;
} MccRiskEntry;

typedef struct {
    MccRiskEntry *entries;
    int count;
} MccRiskTable;

typedef struct {
    NormalizationConfig norm;
    MccRiskTable mcc_table;
} VectorizerContext;

int vectorizer_init(
    VectorizerContext *ctx,
    const char *normalization_path,
    const char *mcc_risk_path
);

void vectorizer_destroy(VectorizerContext *ctx);

int build_transaction_vector(
    VectorizerContext *ctx,
    const char *json_payload,
    float out_vector[VECTOR_SIZE]
);

#endif