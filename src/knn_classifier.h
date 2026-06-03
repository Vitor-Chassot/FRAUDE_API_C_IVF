#ifndef KNN_CLASSIFIER_H
#define KNN_CLASSIFIER_H
typedef struct {
    int approved;
    float fraud_score;
} KnnResult;

int knn_predict(KnnResult *result,const float query[14]);
void load_data(void);
void analyze_features(void);

#endif