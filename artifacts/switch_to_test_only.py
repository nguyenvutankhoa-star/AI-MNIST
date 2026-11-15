"""One-time migration; preserve training sources and produce inference-only C."""
from pathlib import Path

old=Path('Source C/mnist_PTQ.c').read_text()
archive=Path('artifacts/training_sources')
archive.mkdir(exist_ok=True)
for tag in ('PTQ','QAT'):
    dest=archive/f'mnist_{tag}_training.c'
    if not dest.exists():
        dest.write_text(Path(f'Source C/mnist_{tag}.c').read_text())

kernel=old[old.index('/* Symmetric round-to-nearest'):old.index('static int argmax')]
source=r'''/* Test-only C11: evaluate exactly 10000 existing MNIST test images.
 * No training, calibration, float reference pass, or model export.
 * Loads artifacts/@TAG@_fpga.bin, previously trained/quantized on train data.
 * gcc -O3 -std=c11 "Source C/mnist_@TAG@.c" -lm -o artifacts/@TAG@_test.exe
 * Run with no arguments, from project root or Source C or artifacts.
 * BN is already folded into conv weights/bias and activation thresholds.
 * Conv -> threshold -> pool is fused. INT8 data, INT32 MAC, INT64 rescale.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <limits.h>
#define TAG "@TAG@"
#define TEST_COUNT 10000

typedef struct {
    int in, ci, co, h, p, n, ei, ew, eo;
    int8_t qw[4000];
    int32_t qb[16], qt[16], sign[16];
} Layer;
static Layer l[3];
static int8_t qx[784], q1[1352], q2[400];
static const char *root;

static void fail(const char *message) {
    fprintf(stderr, "ERROR: %s\n", message);
    exit(EXIT_FAILURE);
}

static void path_for(char *out, size_t n, const char *relative) {
    if (snprintf(out, n, "%s%s", root, relative) >= (int)n)
        fail("path too long");
}

static FILE *open_relative(const char *relative) {
    char path[1024];
    path_for(path, sizeof path, relative);
    FILE *f = fopen(path, "rb");
    if (!f) fail(path);
    return f;
}

static void locate_root(void) {
    const char *candidates[] = {"", "../"};
    for (unsigned k = 0; k < sizeof candidates / sizeof candidates[0]; k++) {
        char path[1024];
        root = candidates[k];
        path_for(path, sizeof path, "artifacts/" TAG "_fpga.bin");
        FILE *f = fopen(path, "rb");
        if (f) { fclose(f); return; }
    }
    fail("model missing: artifacts/" TAG "_fpga.bin; run from project root, Source C or artifacts");
}

static void read_exact(FILE *f, void *out, size_t bytes) {
    if (fread(out, 1, bytes, f) != bytes) fail("truncated input file");
}

static void load_model(void) {
    static const int expected[3][5] = {
        {28, 1, 8, 13, 72}, {13, 8, 16, 5, 1152}, {0, 400, 10, 0, 4000}
    };
    FILE *f = open_relative("artifacts/" TAG "_fpga.bin");
    char magic[4];
    int32_t version;
    read_exact(f, magic, 4);
    read_exact(f, &version, 4);
    if (memcmp(magic, "MNI8", 4) || version != 1) fail("unsupported model format");
    for (int j = 0; j < 3; j++) {
        Layer *a = l + j;
        int32_t m[8];
        read_exact(f, m, sizeof m);
        for (int k = 0; k < 5; k++)
            if (m[k] != expected[j][k]) fail("model architecture mismatch");
        for (int k = 5; k < 8; k++)
            if (m[k] < -30 || m[k] > 30) fail("invalid scale exponent");
        a->in=m[0]; a->ci=m[1]; a->co=m[2]; a->p=m[3]; a->n=m[4];
        a->h=j<2?a->in-2:0;
        a->ei=m[5]; a->ew=m[6]; a->eo=m[7];
        if (j && a->ei != l[j-1].eo) fail("incompatible activation scales");
        int shift = a->ei + a->ew - a->eo;
        if (j < 2 && (shift < -60 || shift > 30)) fail("invalid requantization shift");
        read_exact(f, a->qw, (size_t)a->n);
        read_exact(f, a->qb, (size_t)a->co*4);
        read_exact(f, a->qt, (size_t)a->co*4);
        read_exact(f, a->sign, (size_t)a->co*4);
        for (int c = 0; c < a->co; c++) {
            /* Worst-case MAC bound, including any possible stored INT8 weight. */
            int64_t bound=(int64_t)(a->n/a->co)*127*128;
            if ((int64_t)a->qb[c]+bound > INT32_MAX ||
                (int64_t)a->qb[c]-bound < INT32_MIN) fail("unsafe accumulator bias");
            if (j < 2 && a->sign[c] != 1 && a->sign[c] != -1)
                fail("invalid activation direction");
        }
    }
    if (fgetc(f) != EOF || ferror(f)) fail("unexpected model data");
    fclose(f);
}

static int quant(float x, int e) {
    double v = round(ldexp((double)x, -e));
    return v > 127 ? 127 : v < -127 ? -127 : (int)v;
}

@KERNEL@

int main(int argc, char **argv) {
    (void)argv;
    if (argc != 1) fail("run without arguments: evaluates exactly 10000 test images");
    uint32_t endian = 1;
    if (sizeof(float) != 4 || sizeof(int32_t) != 4 || *(unsigned char *)&endian != 1)
        fail("little-endian host with 32-bit float/int required");
    setvbuf(stdout, NULL, _IOLBF, 0);
    printf("%s: test-only, 10000 images; loading saved INT8 model...\n", TAG);
    locate_root();
    load_model();
    FILE *images = open_relative("data/test_images.bin");
    FILE *labels = open_relative("data/test_labels.bin");
    char output[1024];
    path_for(output, sizeof output, "artifacts/" TAG "_test_predictions.bin");
    FILE *predictions = fopen(output, "wb");
    if (!predictions) fail(output);
    int correct = 0;
    double start = (double)clock()/CLOCKS_PER_SEC;
    for (int k = 0; k < TEST_COUNT; k++) {
        float image[784];
        int32_t label;
        read_exact(images, image, sizeof image);
        read_exact(labels, &label, sizeof label);
        if (label < 0 || label > 9) fail("label outside 0..9");
        for (int i = 0; i < 784; i++) {
            if (!isfinite(image[i]) || image[i] < 0 || image[i] > 1)
                fail("pixel outside float32 [0,1]");
            qx[i] = (int8_t)quant(image[i], l[0].ei);
        }
        int32_t prediction = infer_int(qx);
        correct += prediction == label;
        if (fwrite(&prediction, 4, 1, predictions) != 1) fail("prediction write failed");
        if ((k+1)%1000 == 0) {
            printf("%s: %d/10000, accuracy=%.4f%%\n", TAG, k+1, 100.0*correct/(k+1));
            fflush(stdout);
        }
    }
    if (fgetc(images) != EOF || fgetc(labels) != EOF || ferror(images) || ferror(labels))
        fail("expected exactly 10000 images and labels");
    fclose(images);
    fclose(labels);
    if (fclose(predictions)) fail("prediction close failed");
    printf("%s test=10000 correct=%d accuracy=%.4f%% seconds=%.3f\n", TAG,
           correct, 100.0*correct/TEST_COUNT, (double)clock()/CLOCKS_PER_SEC-start);
    printf("Predictions: %s\n", output);
    return EXIT_SUCCESS;
}
'''.replace('@KERNEL@',kernel)

# Retain the original train/export generator for reproducibility, but make its
# generated files stay in the archive instead of overwriting the test programs.
generator=Path('artifacts/generate_sources.py')
text=generator.read_text().replace("Path(f'Source C/mnist_{mode}.c')", "Path(f'artifacts/training_sources/mnist_{mode}_training.c')")
generator.write_text(text)
for tag in ('PTQ','QAT'):
    Path(f'Source C/mnist_{tag}.c').write_text(source.replace('@TAG@',tag),encoding='utf-8')
