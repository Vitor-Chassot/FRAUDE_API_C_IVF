#define _POSIX_C_SOURCE 200809L

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#include <errno.h>
#include <stdlib.h>

#include "knn_classifier.h"
#include "payload_vectorizer.h"

#define PORT 9999
#define BUFFER_SIZE 1500
#define QUEUE_SIZE 4096
#define WORKERS 8

static VectorizerContext g_vectorizer;
//static KnnContext g_knn;
static volatile int g_ready = 0;

/* ================= QUEUE SAFE ================= */

typedef struct {
    int fds[QUEUE_SIZE];
    uint32_t head;
    uint32_t tail;
    pthread_mutex_t m;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
} Queue;

static Queue q;

static void q_init() {
    memset(&q, 0, sizeof(q));
    pthread_mutex_init(&q.m, NULL);
    pthread_cond_init(&q.not_empty, NULL);
    pthread_cond_init(&q.not_full, NULL);
}

static void q_push(int fd) {
    pthread_mutex_lock(&q.m);

    while ((q.tail - q.head) >= QUEUE_SIZE) {
        pthread_cond_wait(&q.not_full, &q.m);
    }

    q.fds[q.tail & (QUEUE_SIZE - 1)] = fd;
    q.tail++;

    pthread_cond_signal(&q.not_empty);
    pthread_mutex_unlock(&q.m);
}

static int q_pop() {
    pthread_mutex_lock(&q.m);

    while (q.head == q.tail) {
        pthread_cond_wait(&q.not_empty, &q.m);
    }

    int fd = q.fds[q.head & (QUEUE_SIZE - 1)];
    q.head++;

    pthread_cond_signal(&q.not_full);
    pthread_mutex_unlock(&q.m);

    return fd;
}

/* ================= IO SAFE ================= */

static int read_req(int fd, char *buf) {
    int n = recv(fd, buf, BUFFER_SIZE - 1, 0);
    if (n <= 0) return -1;
    buf[n] = 0;
    return n;
}

static int send_all(int fd, const char *buf, int len) {
    int sent = 0;
    while (sent < len) {
        int n = send(fd, buf + sent, len - sent, MSG_NOSIGNAL);
        if (n <= 0) return -1;
        sent += n;
    }
    return 0;
}

/* ================= LOGIC ================= */

static int analisar(const char *json, char *out) {
    float vec[VECTOR_SIZE];
        
    if (build_transaction_vector(&g_vectorizer, json, vec) != 0)
        return -1;
   
   // fflush(stdout);
    KnnResult r;
    if (knn_predict( &r,vec) != 0)
      return -1;
   

    snprintf(out, 100,
        "{\"approved\":%s,\"fraud_score\":%.4f}",
        r.approved ? "true" : "false",
        r.fraud_score
    );

    return 0;
}

/* ================= WORKER ================= */
static float now_ms(void) {

    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);

    return
        (float)ts.tv_sec * 1000.0
        + (float)ts.tv_nsec / 1000000.0;
}
static void *worker(void *arg) {
    (void)arg;

    char buf[BUFFER_SIZE];
    char res[100];

    while (1) {
        
        int fd = q_pop();

        int n = read_req(fd, buf);
        float tInicio = now_ms();
        if (n > 0 && strncmp(buf, "GET /ready", 10) == 0)
        {
            const char *res;
            int http_code;

            if (g_ready)
            {
                res = "OK";
                http_code = 200;
            }
            else
            {
                res = "NOT_READY";
                http_code = 503;
            }

            char resp[256];
            int len = snprintf(resp, sizeof(resp),
                               "HTTP/1.1 %d %s\r\n"
                               "Content-Type: text/plain\r\n"
                               "Content-Length: %lu\r\n"
                               "Connection: close\r\n"
                               "\r\n"
                               "%s",
                               http_code,
                               g_ready ? "OK" : "Service Unavailable",
                               strlen(res),
                               res);

            send_all(fd, resp, len);
        }
        else if (n > 0 && strncmp(buf, "POST /fraud-score", 17) == 0) {

            char *body = strstr(buf, "\r\n\r\n");
            if (body) {
                body += 4;

                //float t0 = now_ms();
                if (analisar(body, res) != 0) {
                    strcpy(res, "{\"approved\":true,\"fraud_score\":0.0}");
                }
                //float tf = now_ms();
                //float fast_ms_2 = tf - t0;
               //fprintf(stderr, "\nalgoritmo %.3f ms\n", fast_ms_2);
                char resp[512];
                int len = snprintf(resp, sizeof(resp),
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %lu\r\nConnection: close\r\n\r\n%s",
                    strlen(res), res
                );

                send_all(fd, resp, len);
            }
        }
        float tFim=now_ms();
        float tempo_total = tFim - tInicio;
        //fprintf(stderr, "\napi %.3f ms\n", tempo_total);
        close(fd);
    }
}
void *init_thread(void *arg)
{
    fprintf(stderr, "Carregando dados...\n");

    

    load_data();

    q_init();

    g_ready = 1;

    fprintf(stderr, "READY!\n");

    return NULL;
}
/* ================= MAIN ================= */

int main() {
    fprintf(stderr,"=== INICIANDO SERVIDOR ===\n");
    
    signal(SIGPIPE, SIG_IGN);
    
    vectorizer_init(&g_vectorizer,
        "resources/normalization.json",
        "resources/mcc_risk.json");

    g_ready = 0;

    pthread_t init;

    pthread_create(&init, NULL, init_thread, NULL);
    pthread_detach(init);

    /*fprintf(stderr, "Carregando dados\n");
fflush(stderr);

    vectorizer_init(&g_vectorizer,
        "resources/normalization.json",
        "resources/mcc_risk.json");

    
    load_data();
    
    

    g_ready = 1;*/
    q_init();

    pthread_t t[WORKERS];
            
    for (int i = 0; i < WORKERS; i++)
        pthread_create(&t[i], NULL, worker, NULL);

    int s = socket(AF_INET, SOCK_STREAM, 0);

    int opt = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in a = {0};
    a.sin_family = AF_INET;
    a.sin_port = htons(PORT);
    a.sin_addr.s_addr = INADDR_ANY;

    bind(s, (struct sockaddr*)&a, sizeof(a));
    listen(s, 4096);

    while (1) {
        int fd = accept(s, NULL, NULL);
        if (fd >= 0) q_push(fd);
    }
}