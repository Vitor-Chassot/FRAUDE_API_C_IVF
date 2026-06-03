


////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/*
 * knn_classifier_2.c – KNN BRUTO, SIMPLES E LEGÍVEL
 */
/* Carrega o binário (chamado apenas na primeira predição) */



#include "knn_classifier.h"
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <float.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>

#define VECTOR_DIMS 14
#define N_CLUSTERS 350
#define N_PROBES 5
/* Dados estáticos carregados uma vez */
static float (*g_vecs)[16] = NULL;
static uint8_t *g_labels = NULL;
static uint64_t g_count = 0;
static int g_loaded = 0;
typedef struct cluster {

 float centroide[16];

 float (*g_vecs)[16];

 uint8_t *labels;

 uint64_t count;

 uint64_t capacity;

} cluster_t;

cluster_t *clusters;

static inline float dist(
 const float *a,
 const float *b
) {
 __m256 va1 =
 _mm256_load_ps(a);

 __m256 vb1 =
 _mm256_load_ps(b);

 __m256 d1 =
 _mm256_sub_ps(va1, vb1);

 __m256 m1 =
 _mm256_mul_ps(d1, d1);

 __m256 va2 =
 _mm256_load_ps(a + 8);

 __m256 vb2 =
 _mm256_load_ps(b + 8);

 __m256 d2 =
 _mm256_sub_ps(va2, vb2);

 __m256 m2 =
 _mm256_mul_ps(d2, d2);

 __m256 s =
 _mm256_add_ps(m1, m2);

 __m128 lo =
 _mm256_castps256_ps128(s);

 __m128 hi =
 _mm256_extractf128_ps(s, 1);

 __m128 sum =
 _mm_add_ps(lo, hi);

 sum =
 _mm_hadd_ps(sum, sum);

 sum =
 _mm_hadd_ps(sum, sum);

 return _mm_cvtss_f32(sum);
}


void load_data(void) {

 FILE *f = fopen("resources/index.bin", "rb");
 if (!f) {
 return;
 }

 char magic[8];
 uint32_t version;
 uint32_t dims;
 uint64_t count;

 if (fread(magic, 8, 1, f) != 1 ||
 fread(&version, 4, 1, f) != 1 ||
 fread(&dims, 4, 1, f) != 1 ||
 fread(&count, 8, 1, f) != 1) {

 fclose(f);
 return;
 }

 if (dims != VECTOR_DIMS) {
 fclose(f);
 return;
 }

 g_count = count;
 g_loaded = 0;

 g_vecs = NULL;
 g_labels = NULL;

 
 // limpa qualquer carga anterior
 
 if (clusters) {
 for (int i = 0; i < N_CLUSTERS; i++) {
 free(clusters[i].g_vecs);
 free(clusters[i].labels);
 }
 free(clusters);
 clusters = NULL;
 }

 clusters = calloc(N_CLUSTERS, sizeof(cluster_t));
 if (!clusters) {
 fclose(f);
 return;
 }

 for (int i = 0; i < N_CLUSTERS; i++) {
 clusters[i].count = 0;
 clusters[i].capacity = 0;
 clusters[i].g_vecs = NULL;
 clusters[i].labels = NULL;

 for (int j = 0; j < 16; j++) {
 clusters[i].centroide[j] = 0.0f;
 }
 }

 const long data_offset = 8L + 4L + 4L + 8L;

 //
 //=====================================================
 //1) RESERVOIR SAMPLING PARA INIT DOS CENTROIDES
 //=====================================================
 
 float reservoir[N_CLUSTERS][16] __attribute__((aligned(32)));
 uint64_t sampled = 0;

 srand(42);

 if (fseek(f, data_offset, SEEK_SET) != 0) {
 goto fail;
 }

 for (uint64_t i = 0; i < count; i++) {

 float vec[16] __attribute__((aligned(32)));
 uint8_t label;

 if (fread(vec, sizeof(float), VECTOR_DIMS, f) != VECTOR_DIMS ||
 fread(&label, 1, 1, f) != 1) {
 goto fail;
 }

 vec[14] = 0.0f;
 vec[15] = 0.0f;

 if (sampled < N_CLUSTERS) {
 memcpy(reservoir[sampled], vec, sizeof(vec));
 sampled++;
 } else {
 uint64_t rnd =
 (((uint64_t)rand()) << 31) ^
 (uint64_t)rand();

 uint64_t j = rnd % (i + 1);

 if (j < N_CLUSTERS) {
 memcpy(reservoir[j], vec, sizeof(vec));
 }
 }
 }

 if (sampled == 0) {
 goto fail;
 }

 for (int i = 0; i < N_CLUSTERS; i++) {
 int src = (sampled < (uint64_t)N_CLUSTERS)
 ? (i % (int)sampled)
 : i;

 memcpy(clusters[i].centroide, reservoir[src], sizeof(reservoir[src]));
 }

 //
 //=====================================================
 //2) KMEANS SIMPLES EM STREAMING
 //=====================================================
 
 const int KMEANS_ITERS = 2;

 float sums[N_CLUSTERS][16];
 uint32_t counts_cluster[N_CLUSTERS];

 for (int iter = 0; iter < KMEANS_ITERS; iter++) {

 memset(sums, 0, sizeof(sums));
 memset(counts_cluster, 0, sizeof(counts_cluster));

 if (fseek(f, data_offset, SEEK_SET) != 0) {
 goto fail;
 }

 for (uint64_t i = 0; i < count; i++) {

 float vec[16] __attribute__((aligned(32)));
 uint8_t label;

 if (fread(vec, sizeof(float), VECTOR_DIMS, f) != VECTOR_DIMS ||
 fread(&label, 1, 1, f) != 1) {
 goto fail;
 }

 vec[14] = 0.0f;
 vec[15] = 0.0f;

 float menor_distancia = FLT_MAX;
 int ind_cluster = -1;

 for (int j = 0; j < N_CLUSTERS; j++) {
 float distancia = dist(vec, clusters[j].centroide);

 if (distancia < menor_distancia) {
 menor_distancia = distancia;
 ind_cluster = j;
 }
 }

 if (ind_cluster < 0) {
 continue;
 }

 counts_cluster[ind_cluster]++;

 for (int j = 0; j < 16; j++) {
 sums[ind_cluster][j] += vec[j];
 }
 }

 for (int c = 0; c < N_CLUSTERS; c++) {

 if (counts_cluster[c] == 0) {
 continue;
 }

 float inv = 1.0f / (float)counts_cluster[c];

 for (int j = 0; j < 16; j++) {
 clusters[c].centroide[j] = (float)(sums[c][j] * inv);
 }
 }
 }

 //
 //=====================================================
 //3) CONTAR TAMANHO FINAL DE CADA CLUSTER
 //=====================================================
 
 uint64_t final_sizes[N_CLUSTERS];
 memset(final_sizes, 0, sizeof(final_sizes));

 if (fseek(f, data_offset, SEEK_SET) != 0) {
 goto fail;
 }

 for (uint64_t i = 0; i < count; i++) {

 float vec[16] __attribute__((aligned(32)));
 uint8_t label;

 if (fread(vec, sizeof(float), VECTOR_DIMS, f) != VECTOR_DIMS ||
 fread(&label, 1, 1, f) != 1) {
 goto fail;
 }

 vec[14] = 0.0f;
 vec[15] = 0.0f;

 float menor_distancia = FLT_MAX;
 int ind_cluster = -1;

 for (int j = 0; j < N_CLUSTERS; j++) {
 float distancia = dist(vec, clusters[j].centroide);

 if (distancia < menor_distancia) {
 menor_distancia = distancia;
 ind_cluster = j;
 }
 }

 if (ind_cluster >= 0) {
 final_sizes[ind_cluster]++;
 }
 }

 //
 //=====================================================
 //4) ALOCAR EXATO
 //=====================================================
 
 for (int c = 0; c < N_CLUSTERS; c++) {

 clusters[c].count = 0;
 clusters[c].capacity = final_sizes[c] ? final_sizes[c] : 1;

 if (posix_memalign(
 (void **)&clusters[c].g_vecs,
 32,
 clusters[c].capacity * sizeof(*clusters[c].g_vecs)
 ) != 0) {
 goto fail;
 }

 clusters[c].labels =
 malloc(clusters[c].capacity * sizeof(uint8_t));

 if (!clusters[c].labels) {
 goto fail;
 }
 }

 
 //=====================================================
 //5) POPULAR EXATAMENTE
 // =====================================================
 
 uint64_t fill[N_CLUSTERS];
 memset(fill, 0, sizeof(fill));

 if (fseek(f, data_offset, SEEK_SET) != 0) {
 goto fail;
 }

 for (uint64_t i = 0; i < count; i++) {

 float vec[16] __attribute__((aligned(32)));
 uint8_t label;

 if (fread(vec, sizeof(float), VECTOR_DIMS, f) != VECTOR_DIMS ||
 fread(&label, 1, 1, f) != 1) {
 goto fail;
 }

 vec[14] = 0.0f;
 vec[15] = 0.0f;

 float menor_distancia = FLT_MAX;
 int ind_cluster = -1;

 for (int j = 0; j < N_CLUSTERS; j++) {
 float distancia = dist(vec, clusters[j].centroide);

 if (distancia < menor_distancia) {
 menor_distancia = distancia;
 ind_cluster = j;
 }
 }

 if (ind_cluster < 0) {
 continue;
 }

 uint64_t pos = fill[ind_cluster]++;

 if (pos >= clusters[ind_cluster].capacity) {
 goto fail;
 }

 memcpy(clusters[ind_cluster].g_vecs[pos], vec, sizeof(vec));
 clusters[ind_cluster].labels[pos] = label;
 clusters[ind_cluster].count++;
 }

 fclose(f);
 g_loaded = 1;
 fprintf(
 stderr,
 "[KNN] Carregados %lu registros.\n",
 count
 );
 for (int i=0; i<N_CLUSTERS;i++){
    //fprintf(stderr, "%ld\n", clusters[i].count);
    
 }
 return;

fail:
 fclose(f);

 if (clusters) {
 for (int i = 0; i < N_CLUSTERS; i++) {
 free(clusters[i].g_vecs);
 free(clusters[i].labels);
 clusters[i].g_vecs = NULL;
 clusters[i].labels = NULL;
 clusters[i].count = 0;
 clusters[i].capacity = 0;
 }
 free(clusters);
 clusters = NULL;
 }

 g_loaded = 0;
 g_count = 0;
}

static float now_ms(void) {

 struct timespec ts;

 clock_gettime(CLOCK_MONOTONIC, &ts);

 return
 (float)ts.tv_sec * 1000.0
 + (float)ts.tv_nsec / 1000000.0;
}

int knn_predict(KnnResult *result, const float query[14]) {
 //fprintf(stderr, "[KNN] Iniciando predição...\n");
 
 float q[16] __attribute__((aligned(32)));

 for (int i = 0; i < 14; i++)
 {
 q[i] = query[i];
 }

 q[14] = 0.0f;
 q[15] = 0.0f;
 
 // if (!g_loaded) load_data();
 if (!g_loaded) {
 result->approved = 1;
 result->fraud_score = 0;
 return 0;
 }
 
 int k = 5;
 float best_d[k];
 uint8_t best_l[k];
 for (int i = 0; i < k; i++) {
 best_d[i] = 1e30f;
 best_l[i]=0;
 }
 
 
 
 
 int z=0;

 
 //
 //=========================================================
 //ESCOLHER N_PROBES MELHORES CLUSTERS
 // =========================================================
 

int best_clusters[N_PROBES];
float best_cluster_dist[N_PROBES];

for (int i = 0; i < N_PROBES; i++) {

 best_clusters[i] = -1;
 best_cluster_dist[i] = FLT_MAX;
}
//float t0 = now_ms();
for (int i = 0; i < N_CLUSTERS; i++) {

 float *vecs=clusters[i].centroide;
 //if((q[3]-vecs[3])*(q[3]-vecs[3])>best_cluster_dist[N_PROBES - 1])
 //continue;
 
 float d =
 dist(
 q,
 clusters[i].centroide
 );
 

 if (d < best_cluster_dist[N_PROBES - 1]) {

 best_cluster_dist[N_PROBES - 1] = d;

 best_clusters[N_PROBES - 1] = i;

 int j = N_PROBES - 2;

 while (
 j >= 0 &&
 best_cluster_dist[j + 1]
 < best_cluster_dist[j]
 ) {

 float d_aux =
 best_cluster_dist[j];

 int c_aux =
 best_clusters[j];

 best_cluster_dist[j] =
 best_cluster_dist[j + 1];

 best_clusters[j] =
 best_clusters[j + 1];

 best_cluster_dist[j + 1] =
 d_aux;

 best_clusters[j + 1] =
 c_aux;

 j--;
 }
 }
}
//float t1=now_ms();
//fprintf(stderr, "Tempo para procurar nos clusters: %f\n", (t1-t0)/1000.00);
//
 //=========================================================
 //BUSCAR APENAS NOS N_PROBES
 //=========================================================
 //

//float t2=now_ms();

for (int p = 0;
 p < N_PROBES;
 p++) {

 int cid =
 best_clusters[p];

 if (cid < 0)
 continue;

 for (uint64_t i = 0;
 i < clusters[cid].count;
 i++) {
 
 if (i + 16
 < clusters[cid].count) {

 __builtin_prefetch(
 clusters[cid]
 .g_vecs[i + 16],
 0,
 1
 );
 }
 float *vecs=clusters[cid].g_vecs[i];
 //if((q[3]-vecs[3])*(q[3]-vecs[3])>best_d[k-1])
 //continue;
 float d =
 dist(
 q,
 vecs
 );



 if (d < best_d[k - 1]) {

 best_d[k - 1] = d;

 best_l[k - 1] =
 clusters[cid]
 .labels[i];

 int j = k - 2;

 while (
 j >= 0 &&
 best_d[j + 1]
 < best_d[j]
 ) {

 float d_aux =
 best_d[j];

 uint8_t l_aux =
 best_l[j];

 best_d[j] =
 best_d[j + 1];

 best_l[j] =
 best_l[j + 1];

 best_d[j + 1] =
 d_aux;

 best_l[j + 1] =
 l_aux;

 j--;
 }
 }
 }
}
//float t3=now_ms();
//fprintf(stderr,"Busca nos clusters: %f\n", (t3-t2)/1000.00) ;
 
 //fprintf(stderr, "\n tempo que importa: %.3f ms\n", fast_ms);
 
 int fraud = 0;
 

 for (int i = 0; i < k; i++)
 {
 fraud+=best_l[i];
 }
 //fprintf(stderr, "%d", fraud);
 float score = (float)fraud / k;
 result->fraud_score = score;
 result->approved = (fraud < 3);

 return 0;
}



