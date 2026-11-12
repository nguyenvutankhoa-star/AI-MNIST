#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/stat.h>

/*
 * KIEN TRUC (khop voi Training_step.ipynb):
 *   Input -> Conv2D(8,3x3,relu) -> BN -> MaxPool2 -> Conv2D(16,3x3,relu) -> BN
 *         -> MaxPool2 -> Flatten -> Dense(10)        (Dropout tat khi suy luan)
 * BatchNorm nam SAU ReLU nen khong gop vao Conv duoc -> tinh rieng (scale/shift).
 * Keras BatchNormalization mac dinh epsilon = 1e-3.
 */
#define IMG        28
#define NUM_CLASS  10
#define C1         8
#define C2         16
#define BN_EPS     1e-3f

/* Cac thu muc tim file, thu lan luot (tuyet doi truoc, tuong doi sau) */
static const char *SEARCH_DIRS[] = {
    "D:/reports/weights/", "D:/reports/data/",
    "../weights/", "../data/", "weights/", "data/", "./"
};
#define N_DIRS ((int)(sizeof(SEARCH_DIRS) / sizeof(SEARCH_DIRS[0])))

static void die(const char *msg, const char *arg) {
    fprintf(stderr, "LOI: %s %s\n", msg, arg ? arg : "");
    exit(1);
}

/*
 * Tim file theo ten. Thu ca ten dung va ten co dau cach o dau (vd " bn2_gamma.txt",
 * do code export trong notebook luu nham). Neu co nhieu ban, chon ban MOI NHAT.
 * Tra ve 1 neu tim thay.
 */
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

/* Doc toan bo so thuc trong file text (cach nhau boi khoang trang/xuong dong/dau phay) */
static float *read_text_floats(const char *path, size_t *count) {
    FILE *f = fopen(path, "r");
    if (!f) return NULL;
    size_t cap = 1024, n = 0;
    float *buf = (float *)malloc(cap * sizeof(float));
    char tok[64];
    int c, len = 0;
    for (;;) {
        c = fgetc(f);
        int sep = (c == EOF || c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == ',');
        if (sep) {
            if (len > 0) {
                tok[len] = 0;
                if (n == cap) { cap *= 2; buf = (float *)realloc(buf, cap * sizeof(float)); }
                buf[n++] = (float)atof(tok);
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

/* Doc file nhi phan -> buffer (so byte tra ve qua nbytes) */
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

/* Doc 1 tham so; expected = 0 nghia la khong kiem tra so phan tu */
static float *load_param(const char *name, size_t expected, size_t *got) {
    char path[600];
    if (!find_file(name, path, sizeof(path))) die("Khong tim thay file", name);
    size_t n = 0;
    float *a = read_text_floats(path, &n);
    if (!a) die("Khong mo duoc file", path);
    if (expected && n != expected) {
        fprintf(stderr, "LOI: %s co %zu gia tri, can %zu (file cu/sai kien truc?)\n", path, n, expected);
        exit(1);
    }
    printf("   %-18s <- %s\n", name, path);
    if (got) *got = n;
    return a;
}

/* Tinh scale/shift cua BN: y = x*scale + shift */
static void load_bn(const char *prefix, int ch, float *scale, float *shift) {
    char nm[64];
    snprintf(nm, sizeof(nm), "%s_gamma.txt", prefix); float *g = load_param(nm, ch, NULL);
    snprintf(nm, sizeof(nm), "%s_beta.txt",  prefix); float *b = load_param(nm, ch, NULL);
    snprintf(nm, sizeof(nm), "%s_mean.txt",  prefix); float *m = load_param(nm, ch, NULL);
    snprintf(nm, sizeof(nm), "%s_var.txt",   prefix); float *v = load_param(nm, ch, NULL);
    for (int i = 0; i < ch; i++) {
        scale[i] = g[i] / sqrtf(v[i] + BN_EPS);
        shift[i] = b[i] - m[i] * scale[i];
    }
    free(g); free(b); free(m); free(v);
}

typedef struct {
    int in, h1, p1, h2, p2, dense_in;   /* kich thuoc cac tang */
    float *w1, *b1, sc1[C1], sh1[C1];   /* w1: [kh][kw][1][8]   */
    float *w2, *b2, sc2[C2], sh2[C2];   /* w2: [kh][kw][8][16]  */
    float *wd, *bd;                     /* wd: [dense_in][10]   */
} Model;

/* Suy luan cho MOT anh (x: in*in, da padding neu can). Cac buffer trung gian duoc truyen vao. */
static void infer_single_c(const Model *m, const float *x,
                           float *c1, float *pl1, float *c2, float *pl2, float *logits)
{
    const int in = m->in, h1 = m->h1, p1 = m->p1, h2 = m->h2, p2 = m->p2;

    /* 1. Conv1 + ReLU + BN */
    for (int i = 0; i < h1; i++) {
        for (int j = 0; j < h1; j++) {
            float acc[C1];
            for (int f = 0; f < C1; f++) acc[f] = m->b1[f];
            for (int ki = 0; ki < 3; ki++)
                for (int kj = 0; kj < 3; kj++) {
                    float px = x[(i + ki) * in + (j + kj)];
                    const float *w = m->w1 + (ki * 3 + kj) * C1;
                    for (int f = 0; f < C1; f++) acc[f] += px * w[f];
                }
            float *o = c1 + (i * h1 + j) * C1;
            for (int f = 0; f < C1; f++) {
                float r = acc[f] > 0.0f ? acc[f] : 0.0f;
                o[f] = r * m->sc1[f] + m->sh1[f];
            }
        }
    }

    /* 2. MaxPool 2x2 (bo hang/cot le cuoi giong Keras 'valid') */
    for (int i = 0; i < p1; i++)
        for (int j = 0; j < p1; j++)
            for (int c = 0; c < C1; c++) {
                float a = c1[((2 * i)     * h1 + 2 * j)     * C1 + c];
                float b = c1[((2 * i)     * h1 + 2 * j + 1) * C1 + c];
                float d = c1[((2 * i + 1) * h1 + 2 * j)     * C1 + c];
                float e = c1[((2 * i + 1) * h1 + 2 * j + 1) * C1 + c];
                float m1 = a > b ? a : b, m2 = d > e ? d : e;
                pl1[(i * p1 + j) * C1 + c] = m1 > m2 ? m1 : m2;
            }

    /* 3. Conv2 + ReLU + BN */
    for (int i = 0; i < h2; i++) {
        for (int j = 0; j < h2; j++) {
            float acc[C2];
            for (int f = 0; f < C2; f++) acc[f] = m->b2[f];
            for (int ki = 0; ki < 3; ki++)
                for (int kj = 0; kj < 3; kj++)
                    for (int c = 0; c < C1; c++) {
                        float v = pl1[((i + ki) * p1 + (j + kj)) * C1 + c];
                        const float *w = m->w2 + ((ki * 3 + kj) * C1 + c) * C2;
                        for (int f = 0; f < C2; f++) acc[f] += v * w[f];
                    }
            float *o = c2 + (i * h2 + j) * C2;
            for (int f = 0; f < C2; f++) {
                float r = acc[f] > 0.0f ? acc[f] : 0.0f;
                o[f] = r * m->sc2[f] + m->sh2[f];
            }
        }
    }

    /* 4. MaxPool 2x2 */
    for (int i = 0; i < p2; i++)
        for (int j = 0; j < p2; j++)
            for (int c = 0; c < C2; c++) {
                float a = c2[((2 * i)     * h2 + 2 * j)     * C2 + c];
                float b = c2[((2 * i)     * h2 + 2 * j + 1) * C2 + c];
                float d = c2[((2 * i + 1) * h2 + 2 * j)     * C2 + c];
                float e = c2[((2 * i + 1) * h2 + 2 * j + 1) * C2 + c];
                float m1 = a > b ? a : b, m2 = d > e ? d : e;
                pl2[(i * p2 + j) * C2 + c] = m1 > m2 ? m1 : m2;
            }

    /* 5. Flatten (thu tu H,W,C) + Dense */
    for (int o = 0; o < NUM_CLASS; o++) logits[o] = m->bd[o];
    for (int k = 0; k < m->dense_in; k++) {
        float v = pl2[k];
        const float *w = m->wd + k * NUM_CLASS;
        for (int o = 0; o < NUM_CLASS; o++) logits[o] += v * w[o];
    }
}

int main(void) {
    printf("Dang tim va tai cac file trong so (uu tien %s)\n", SEARCH_DIRS[0]);

    Model m;
    memset(&m, 0, sizeof(m));

    /* ---- Trong so ---- */
    m.w1 = load_param("conv1_weights.txt", 3 * 3 * 1 * C1, NULL);
    m.b1 = load_param("conv1_bias.txt", C1, NULL);
    load_bn("bn1", C1, m.sc1, m.sh1);

    m.w2 = load_param("conv2_weights.txt", 3 * 3 * C1 * C2, NULL);
    m.b2 = load_param("conv2_bias.txt", C2, NULL);
    load_bn("bn2", C2, m.sc2, m.sh2);

    size_t nd = 0;
    m.wd = load_param("dense_weights.txt", 0, &nd);
    m.bd = load_param("dense_bias.txt", NUM_CLASS, NULL);
    if (nd % NUM_CLASS) die("dense_weights.txt khong chia het cho 10", NULL);
    m.dense_in = (int)(nd / NUM_CLASS);

    /* Suy ra kich thuoc anh vao mang (co/khong padding) tu so dau vao cua Dense */
    int pad = -1;
    for (int p = 0; p < 4 && pad < 0; p++) {
        int s = IMG + 2 * p;
        int h1 = s - 2, p1 = h1 / 2, h2 = p1 - 2, p2 = h2 / 2;
        if (p2 * p2 * C2 == m.dense_in) {
            pad = p;
            m.in = s; m.h1 = h1; m.p1 = p1; m.h2 = h2; m.p2 = p2;
        }
    }
    if (pad < 0) die("So dau vao cua Dense khong khop kien truc (400 hoac 576)", NULL);
    printf("-> Dense: %d dau vao => anh vao mang %dx%d (padding=%d)\n",
           m.dense_in, m.in, m.in, pad);

    /* ---- Du lieu test ---- */
    char pi[600], pl[600];
    float *img_raw = NULL;
    int *labels = NULL;
    size_t n_img_vals = 0, n_lbl = 0;

    if (find_file("test_images.bin", pi, sizeof(pi)) && find_file("test_labels.bin", pl, sizeof(pl))) {
        size_t b1 = 0, b2 = 0;
        img_raw = (float *)read_binary(pi, &b1);
        labels  = (int *)read_binary(pl, &b2);
        if (!img_raw || !labels) die("Khong doc duoc file .bin", NULL);
        n_img_vals = b1 / sizeof(float);
        n_lbl = b2 / sizeof(int);
        printf("-> Da tai du lieu test tu %s\n", pi);
    } else if (find_file("test_images.txt", pi, sizeof(pi)) && find_file("test_labels.txt", pl, sizeof(pl))) {
        float *lf = read_text_floats(pl, &n_lbl);
        img_raw = read_text_floats(pi, &n_img_vals);
        if (!img_raw || !lf) die("Khong doc duoc file .txt", NULL);
        labels = (int *)malloc(n_lbl * sizeof(int));
        for (size_t i = 0; i < n_lbl; i++) labels[i] = (int)(lf[i] + 0.5f);
        free(lf);
        printf("-> Da tai du lieu test tu %s\n", pi);
    } else {
        die("Khong tim thay test_images.bin/test_labels.bin (hoac .txt). "
            "Hay chay infer_python.py truoc de no xuat 2 file .bin vao D:\\reports\\data", NULL);
    }

    const int N = (int)n_lbl;
    if (N <= 0 || n_img_vals % (size_t)N) die("So anh va so nhan khong khop", NULL);
    int side = (int)(n_img_vals / (size_t)N);   /* so gia tri / anh: 784 hoac 900 (da padding) */
    int s = (int)(sqrt((double)side) + 0.5);
    if (s * s != side || s < IMG) die("Kich thuoc moi anh khong hop le", NULL);
    int crop = (s - IMG) / 2;

    /* Chuan bi anh dua vao mang: cat vien (neu file da padding), chuan hoa 0..1, roi pad dung kich thuoc */
    float vmax = 0.0f;
    for (size_t i = 0; i < n_img_vals; i++) if (img_raw[i] > vmax) vmax = img_raw[i];
    float norm = vmax > 1.5f ? 1.0f / 255.0f : 1.0f;

    const size_t img_sz = (size_t)m.in * m.in;
    float *x_batch = (float *)calloc((size_t)N * img_sz, sizeof(float));
    float *logits_batch = (float *)malloc((size_t)N * NUM_CLASS * sizeof(float));
    for (int b = 0; b < N; b++)
        for (int i = 0; i < IMG; i++)
            for (int j = 0; j < IMG; j++)
                x_batch[b * img_sz + (size_t)(i + pad) * m.in + (j + pad)] =
                    img_raw[(size_t)b * side + (size_t)(i + crop) * s + (j + crop)] * norm;
    free(img_raw);

    /* Buffer trung gian (cap phat 1 lan, ngoai vung do thoi gian) */
    float *c1  = (float *)malloc((size_t)m.h1 * m.h1 * C1 * sizeof(float));
    float *pl1 = (float *)malloc((size_t)m.p1 * m.p1 * C1 * sizeof(float));
    float *c2  = (float *)malloc((size_t)m.h2 * m.h2 * C2 * sizeof(float));
    float *pl2 = (float *)malloc((size_t)m.dense_in * sizeof(float));

    /* Lam nong (warm-up) */
    for (int b = 0; b < (N < 10 ? N : 10); b++)
        infer_single_c(&m, x_batch + b * img_sz, c1, pl1, c2, pl2, logits_batch + b * NUM_CLASS);

    /* --- BAT DAU DO THOI GIAN SUY LUAN --- */
    clock_t start = clock();
    for (int b = 0; b < N; b++)
        infer_single_c(&m, x_batch + b * img_sz, c1, pl1, c2, pl2, logits_batch + b * NUM_CLASS);
    clock_t end = clock();
    /* --- KET THUC DO THOI GIAN --- */

    /* Do chinh xac (Argmax) */
    int correct = 0;
    for (int b = 0; b < N; b++) {
        float mx = logits_batch[b * NUM_CLASS];
        int pred = 0;
        for (int i = 1; i < NUM_CLASS; i++)
            if (logits_batch[b * NUM_CLASS + i] > mx) { mx = logits_batch[b * NUM_CLASS + i]; pred = i; }
        if (pred == labels[b]) correct++;
    }
    int incorrect = N - correct;
    double accuracy = (double)correct / N * 100.0;
    double time_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;

    printf("\n========== BAO CAO C (GOLDEN MODEL) ==========\n");
    printf("Tong so anh kiem tra : %d anh\n", N);
    printf("Du doan DUNG         : %d anh\n", correct);
    printf("Du doan SAI          : %d anh\n", incorrect);
    printf("Ti le chinh xac      : %.2f%%\n", accuracy);
    printf("--------------------------------------------\n");
    printf("Tong thoi gian chay  : %.2f ms\n", time_ms);
    printf("Thoi gian/1 anh      : %.5f ms\n", time_ms / N);
    printf("==============================================\n");

    free(x_batch); free(logits_batch); free(labels);
    free(c1); free(pl1); free(c2); free(pl2);
    free(m.w1); free(m.b1); free(m.w2); free(m.b2); free(m.wd); free(m.bd);
    return 0;
}