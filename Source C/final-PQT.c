#include <stdio.h>
#include <stdlib.h>
#include <stdint.h> /* Bat buoc de dung cac kieu so nguyen int8_t, int16_t, int32_t */
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#inlcude <math.h>

/*
 * KIEN TRUC (Lượng hóa - INT8):
 * Input (INT8) -> Conv2D_1 (INT8) -> Multiply_Shift_Add (INT32) -> ReLU (INT8) 
 *              -> MaxPool2 (INT8) -> Conv2D_2 (INT8) -> Multiply_Shift_Add (INT32)
 *              -> ReLU (INT8) -> MaxPool2 (INT8) -> Flatten -> Dense (INT8)
 *
 * YEU CAU QUAN TRONG:
 * - Tat ca he so (Weights) tu Float phai duoc luong hoa thanh INT8 qua Python.
 * - Batch Normalization phai duoc gop vao he so cua Conv2D hoac chuyen thanh phan so (Multiplier / Shift).
 * - Code nay mo phong logic FPGA voi phep tinh dich bit (Bit-shifting) cho toc do nhanh nhat.
 */

#define IMG        28
#define NUM_CLASS  10
#define C1         8
#define C2         16

static const char *SEARCH_DIRS[] = {
    "D:/reports/weights_quantized/", "D:/reports/data/",
    "../weights_quantized/", "../data/", "weights_quantized/", "data/", "./"
};
#define N_DIRS ((int)(sizeof(SEARCH_DIRS) / sizeof(SEARCH_DIRS[0])))

static void die(const char *msg, const char *arg) {
    fprintf(stderr, "LOI: %s %s\n", msg, arg ? arg : "");
    exit(1);
}

/* Ham ho tro doc du lieu nguyen thuy tuong tu ban truoc... */
static int find_file(const char *name, char *out, size_t out_sz) {
    char cand[600];
    time_t best_t = 0;
    int found = 0;
    for (int d = 0; d < N_DIRS; d++) {
        for (int v = 0; v < 2; v++) {
            snprintf(cand, sizeof(cand), "%s%s%s", SEARCH_DIRS[d], v ? " " : "", name);
            struct stat st;
            if (stat(cand, &st) == 0 && (!found || st.st_mtime > best_t)) {
                best_t = st.st_mtime;
                snprintf(out, out_sz, "%s", cand);
                found = 1;
            }
        }
    }
    return found;
}

/* Doc file text ra mang int32 (cho viec nap he so tu Python export) */
static int32_t *read_text_int32(const char *path, size_t *count) {
    FILE *f = fopen(path, "r");
    if (!f) return NULL;
    size_t cap = 1024, n = 0;
    int32_t *buf = (int32_t *)malloc(cap * sizeof(int32_t));
    char tok[64];
    int c, len = 0;
    for (;;) {
        c = fgetc(f);
        int sep = (c == EOF || c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == ',');
        if (sep) {
            if (len > 0) {
                tok[len] = 0;
                if (n == cap) { cap *= 2; buf = (int32_t *)realloc(buf, cap * sizeof(int32_t)); }
                buf[n++] = (int32_t)atoi(tok); /* Doi tu Float -> Int */
                len = 0;
            }
            if (c == EOF) break;
        } else if (len < (int)sizeof(tok) - 1) {
            tok[len++] = (char)c;
        }
    }
    fclose(f);
    *count = n;
    return buf;
}

/* Doc 1 tham so da luong hoa; expected = 0 nghia la khong kiem tra so phan tu */
static int32_t *load_param_int32(const char *name, size_t expected, size_t *got) {
    char path[600];
    if (!find_file(name, path, sizeof(path))) die("Khong tim thay file luong hoa", name);
    size_t n = 0;
    int32_t *a = read_text_int32(path, &n);
    if (!a) die("Khong mo duoc file", path);
    if (expected && n != expected) {
        fprintf(stderr, "LOI: %s co %zu gia tri, can %zu (sai kien truc?)\n", path, n, expected);
        exit(1);
    }
    printf("   %-18s <- %s\n", name, path);
    if (got) *got = n;
    return a;
}

static void *read_binary(const char *path, size_t *nbytes) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *buf = malloc((size_t)sz);
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    if (rd != (size_t)sz) { free(buf); return NULL; }
    *nbytes = (size_t)sz;
    return buf;
}

/* Kieu du lieu Model dung hoan toan so nguyen */
typedef struct {
    int in, h1, p1, h2, p2, dense_in;   
    int8_t *w1;         /* [kh][kw][1][8]   */
    int32_t *b1;        
    int32_t *mult1, *shift1; /* He so chia ty le INT thay cho sc1, sh1 */
    
    int8_t *w2;         /* [kh][kw][8][16]  */
    int32_t *b2;
    int32_t *mult2, *shift2; 
    
    int8_t *wd;         
    int32_t *bd;        
} QuantizedModel;

/*
 * Phep tinh Fixed-Point tuong duong phep nhan so thuc.
 * Su dung tren nhung kien truc FPGA ho tro phep nhan so nguyen dai hoac dich bit.
 */
static int8_t multiply_by_quantized_multiplier(int32_t x, int32_t quantized_multiplier, int shift) {
    int64_t round = (1LL << (shift - 1));
    int64_t result = ((int64_t)x * (int64_t)quantized_multiplier + round) >> shift;
    
    /* Ham luong (Clipping) cho so INT8 [-128, 127] */
    if (result > 127) return 127;
    if (result < -128) return -128;
    return (int8_t)result;
}

/* Ham ReLU Fixed-point (loai bo so am) */
static inline int8_t relu_int8(int8_t x) {
    return (x > 0) ? x : 0;
}

/* Suy luan Quantized Model cho 1 anh, xu ly tuyet doi bang so nguyen */
static void infer_single_quantized(const QuantizedModel *m, const int8_t *x,
                           int8_t *c1, int8_t *pl1, int8_t *c2, int8_t *pl2, int32_t *logits)
{
    const int in = m->in, h1 = m->h1, p1 = m->p1, h2 = m->h2, p2 = m->p2;

    /* 1. Conv1 + Fixed-Point BN + ReLU */
    for (int i = 0; i < h1; i++) {
        for (int j = 0; j < h1; j++) {
            int32_t acc[C1]; /* Thanh ghi tich luy kieu rong */
            for (int f = 0; f < C1; f++) acc[f] = m->b1[f];
            
            for (int ki = 0; ki < 3; ki++)
                for (int kj = 0; kj < 3; kj++) {
                    int8_t px = x[(i + ki) * in + (j + kj)];
                    const int8_t *w = m->w1 + (ki * 3 + kj) * C1;
                    for (int f = 0; f < C1; f++) acc[f] += px * w[f];
                }
                
            int8_t *o = c1 + (i * h1 + j) * C1;
            for (int f = 0; f < C1; f++) {
                int8_t scaled = multiply_by_quantized_multiplier(acc[f], m->mult1[f], m->shift1[f]);
                o[f] = relu_int8(scaled);
            }
        }
    }

    /* 2. MaxPool 2x2 INT8 */
    for (int i = 0; i < p1; i++)
        for (int j = 0; j < p1; j++)
            for (int c = 0; c < C1; c++) {
                int8_t a = c1[((2 * i)     * h1 + 2 * j)     * C1 + c];
                int8_t b = c1[((2 * i)     * h1 + 2 * j + 1) * C1 + c];
                int8_t d = c1[((2 * i + 1) * h1 + 2 * j)     * C1 + c];
                int8_t e = c1[((2 * i + 1) * h1 + 2 * j + 1) * C1 + c];
                int8_t m1 = a > b ? a : b, m2 = d > e ? d : e;
                pl1[(i * p1 + j) * C1 + c] = m1 > m2 ? m1 : m2;
            }

    /* 3. Conv2 + Fixed-Point BN + ReLU */
    for (int i = 0; i < h2; i++) {
        for (int j = 0; j < h2; j++) {
            int32_t acc[C2];
            for (int f = 0; f < C2; f++) acc[f] = m->b2[f];
            
            for (int ki = 0; ki < 3; ki++)
                for (int kj = 0; kj < 3; kj++)
                    for (int c = 0; c < C1; c++) {
                        int8_t v = pl1[((i + ki) * p1 + (j + kj)) * C1 + c];
                        const int8_t *w = m->w2 + ((ki * 3 + kj) * C1 + c) * C2;
                        for (int f = 0; f < C2; f++) acc[f] += v * w[f];
                    }
                    
            int8_t *o = c2 + (i * h2 + j) * C2;
            for (int f = 0; f < C2; f++) {
                int8_t scaled = multiply_by_quantized_multiplier(acc[f], m->mult2[f], m->shift2[f]);
                o[f] = relu_int8(scaled);
            }
        }
    }

    /* 4. MaxPool 2x2 INT8 */
    for (int i = 0; i < p2; i++)
        for (int j = 0; j < p2; j++)
            for (int c = 0; c < C2; c++) {
                int8_t a = c2[((2 * i)     * h2 + 2 * j)     * C2 + c];
                int8_t b = c2[((2 * i)     * h2 + 2 * j + 1) * C2 + c];
                int8_t d = c2[((2 * i + 1) * h2 + 2 * j)     * C2 + c];
                int8_t e = c2[((2 * i + 1) * h2 + 2 * j + 1) * C2 + c];
                int8_t m1 = a > b ? a : b, m2 = d > e ? d : e;
                pl2[(i * p2 + j) * C2 + c] = m1 > m2 ? m1 : m2;
            }

    /* 5. Flatten + Dense INT8 tich luy tren INT32 */
    for (int o = 0; o < NUM_CLASS; o++) logits[o] = m->bd[o];
    for (int k = 0; k < m->dense_in; k++) {
        int8_t v = pl2[k];
        const int8_t *w = m->wd + k * NUM_CLASS;
        for (int o = 0; o < NUM_CLASS; o++) logits[o] += v * w[o];
    }
}

/* Ham cast cac mang Int32 ve Int8 (vi file text doc ra kieu Int32) */
static int8_t* cast_to_int8(int32_t* src, size_t n) {
    int8_t *dst = (int8_t *)malloc(n * sizeof(int8_t));
    for (size_t i = 0; i < n; i++) dst[i] = (int8_t)src[i];
    free(src);
    return dst;
}

int main(void) {
    printf("Dang tai he so LUONG HOA (PTQ INT8) tu Python...\n");

    QuantizedModel m;
    memset(&m, 0, sizeof(m));

    /* --- Luong hoa he so --- */
    size_t tmp_n;
    m.w1 = cast_to_int8(load_param_int32("q_conv1_w.txt", 3 * 3 * 1 * C1, &tmp_n), tmp_n);
    m.b1 = load_param_int32("q_conv1_b.txt", C1, NULL);
    m.mult1 = load_param_int32("q_bn1_mult.txt", C1, NULL);
    m.shift1 = load_param_int32("q_bn1_shift.txt", C1, NULL);

    m.w2 = cast_to_int8(load_param_int32("q_conv2_w.txt", 3 * 3 * C1 * C2, &tmp_n), tmp_n);
    m.b2 = load_param_int32("q_conv2_b.txt", C2, NULL);
    m.mult2 = load_param_int32("q_bn2_mult.txt", C2, NULL);
    m.shift2 = load_param_int32("q_bn2_shift.txt", C2, NULL);

    size_t nd = 0;
    m.wd = cast_to_int8(load_param_int32("q_dense_w.txt", 0, &nd), nd);
    m.bd = load_param_int32("q_dense_b.txt", NUM_CLASS, NULL);
    
    if (nd % NUM_CLASS) die("q_dense_w.txt khong chia het cho 10", NULL);
    m.dense_in = (int)(nd / NUM_CLASS);

    /* Kiem tra kich thuoc */
    int pad = -1;
    for (int p = 0; p < 4 && pad < 0; p++) {
        int s = IMG + 2 * p;
        int h1 = s - 2, p1 = h1 / 2, h2 = p1 - 2, p2 = h2 / 2;
        if (p2 * p2 * C2 == m.dense_in) {
            pad = p;
            m.in = s; m.h1 = h1; m.p1 = p1; m.h2 = h2; m.p2 = p2;
        }
    }
    if (pad < 0) die("Khong mo phong duoc kien truc Dense", NULL);

    /* --- Du lieu test (phai la INT8 roi tu file bin) --- */
    char pi[600], pl[600];
    int8_t *img_raw = NULL;
    int *labels = NULL;
    size_t n_img_vals = 0, n_lbl = 0;

    /* Gia dinh test_images_int8.bin chua pixel mang gia tri -128 toi 127 */
    if (find_file("test_images_int8.bin", pi, sizeof(pi)) && find_file("test_labels.bin", pl, sizeof(pl))) {
        size_t b1 = 0, b2 = 0;
        img_raw = (int8_t *)read_binary(pi, &b1);
        labels  = (int *)read_binary(pl, &b2);
        n_img_vals = b1;
        n_lbl = b2 / sizeof(int);
    } else {
        die("Vui long chay xuat du lieu luong hoa INT8 tren Python (test_images_int8.bin)!", NULL);
    }

    const int N = (int)n_lbl;
    int side = (int)(n_img_vals / (size_t)N);
    int s = (int)(sqrt((double)side) + 0.5);
    int crop = (s - IMG) / 2;

    const size_t img_sz = (size_t)m.in * m.in;
    int8_t *x_batch = (int8_t *)calloc((size_t)N * img_sz, sizeof(int8_t));
    int32_t *logits_batch = (int32_t *)malloc((size_t)N * NUM_CLASS * sizeof(int32_t));
    
    for (int b = 0; b < N; b++)
        for (int i = 0; i < IMG; i++)
            for (int j = 0; j < IMG; j++)
                x_batch[b * img_sz + (size_t)(i + pad) * m.in + (j + pad)] =
                    img_raw[(size_t)b * side + (size_t)(i + crop) * s + (j + crop)];
    free(img_raw);

    /* Buffer INT8 trung gian */
    int8_t *c1  = (int8_t *)malloc((size_t)m.h1 * m.h1 * C1 * sizeof(int8_t));
    int8_t *pl1 = (int8_t *)malloc((size_t)m.p1 * m.p1 * C1 * sizeof(int8_t));
    int8_t *c2  = (int8_t *)malloc((size_t)m.h2 * m.h2 * C2 * sizeof(int8_t));
    int8_t *pl2 = (int8_t *)malloc((size_t)m.dense_in * sizeof(int8_t));

    /* --- BAT DAU DO THOI GIAN INT8 FPGA SIMULATION --- */
    clock_t start = clock();
    for (int b = 0; b < N; b++)
        infer_single_quantized(&m, x_batch + b * img_sz, c1, pl1, c2, pl2, logits_batch + b * NUM_CLASS);
    clock_t end = clock();
    /* --- KET THUC --- */

    /* Kiem chung kieu Integer ArgMax tren LOGITS INT32 */
    int correct = 0;
    for (int b = 0; b < N; b++) {
        int32_t mx = logits_batch[b * NUM_CLASS];
        int pred = 0;
        for (int i = 1; i < NUM_CLASS; i++)
            if (logits_batch[b * NUM_CLASS + i] > mx) { mx = logits_batch[b * NUM_CLASS + i]; pred = i; }
        if (pred == labels[b]) correct++;
    }
    int incorrect = N - correct;
    double accuracy = (double)correct / N * 100.0;
    double time_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;

    printf("\n========== BAO CAO FPGA HLS (QUANTIZED INT8) ==========\n");
    printf("Tong so anh kiem tra : %d anh\n", N);
    printf("Du doan DUNG         : %d anh\n", correct);
    printf("Du doan SAI          : %d anh\n", incorrect);
    printf("Ti le chinh xac      : %.2f%%\n", accuracy);
    printf("------------------------------------------------------\n");
    printf("Thoi gian mo phong C : %.2f ms\n", time_ms);
    printf("Thoi gian / 1 anh    : %.5f ms\n", time_ms / N);
    printf("======================================================\n");

    free(x_batch); free(logits_batch); free(labels);
    free(c1); free(pl1); free(c2); free(pl2);
    free(m.w1); free(m.b1); free(m.mult1); free(m.shift1);
    free(m.w2); free(m.b2); free(m.mult2); free(m.shift2);
    free(m.wd); free(m.bd);
    return 0;
}