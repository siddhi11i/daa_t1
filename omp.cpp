/*
 * omp.cpp  --  Step 4: OpenMP parallel brute-force SSD template matching
 *
 * Parallelism
 * -----------
 *   Outer loop over images is parallelised with OpenMP.
 *   Each thread processes a disjoint subset of images independently.
 *   The inner sliding-window loop runs sequentially per image.
 *
 * Build
 * -----
 *   g++ -O3 -fopenmp -o omp omp.cpp
 *
 * Usage
 * -----
 *   ./omp <images.bin> <template.bin> <N> <output.txt> [threads] [schedule]
 *
 *   threads  : number of OpenMP threads (default: OMP_NUM_THREADS / hw)
 *   schedule : static | dynamic | guided  (default: dynamic)
 *
 * Timing
 * ------
 *   File-loading time and compute time printed via omp_get_wtime.
 *
 * Output format (identical to seq.cpp)
 * --------------------------------------
 *   image_id x y score
 */

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>
#include <cassert>
#include <string>
#include <omp.h>

static const int IMG_H  = 256;
static const int IMG_W  = 256;
static const int TMPL_H = 32;
static const int TMPL_W = 32;
static const int IMG_PX   = IMG_H * IMG_W;
static const int TMPL_PX  = TMPL_H * TMPL_W;

/* Compute SSD at window position (y, x) inside img. */
static inline int32_t ssd_at(const uint8_t *img,
                              const uint8_t *tmpl,
                              int y, int x)
{
    int32_t s = 0;
    for (int ty = 0; ty < TMPL_H; ++ty) {
        const uint8_t *ri = img  + (y + ty) * IMG_W + x;
        const uint8_t *rt = tmpl + ty * TMPL_W;
        for (int tx = 0; tx < TMPL_W; ++tx) {
            int d = (int)ri[tx] - (int)rt[tx];
            s += d * d;
        }
    }
    return s;
}

int main(int argc, char *argv[])
{
    if (argc < 5) {
        fprintf(stderr,
            "Usage: %s <images.bin> <template.bin> <N> <output.txt> [threads] [schedule]\n",
            argv[0]);
        return 1;
    }

    const char *img_path  = argv[1];
    const char *tmpl_path = argv[2];
    int         N         = atoi(argv[3]);
    const char *out_path  = argv[4];

    /* Optional thread count */
    if (argc >= 6) omp_set_num_threads(atoi(argv[5]));

    /* Optional schedule (static/dynamic/guided), default dynamic */
    std::string sched = (argc >= 7) ? argv[6] : "dynamic";

    /* ---- load template ----------------------------------------------------- */
    double t_load0 = omp_get_wtime();

    FILE *ft = fopen(tmpl_path, "rb");
    if (!ft) { perror(tmpl_path); return 1; }
    uint8_t tmpl[TMPL_PX];
    if (fread(tmpl, 1, TMPL_PX, ft) != (size_t)TMPL_PX) {
        fprintf(stderr, "Template file too small\n"); return 1;
    }
    fclose(ft);

    /* ---- load all images into memory --------------------------------------- */
    FILE *fi = fopen(img_path, "rb");
    if (!fi) { perror(img_path); return 1; }

    std::vector<uint8_t> all_imgs((size_t)N * IMG_PX);
    if (fread(all_imgs.data(), 1, (size_t)N * IMG_PX, fi) != (size_t)N * IMG_PX) {
        fprintf(stderr, "Failed to read all images\n"); return 1;
    }
    fclose(fi);

    double t_load1 = omp_get_wtime();
    printf("Load time    : %.3f s\n", t_load1 - t_load0);

    /* ---- result buffers --------------------------------------------------- */
    std::vector<int>     res_x(N), res_y(N);
    std::vector<int32_t> res_score(N);

    printf("Matching with OpenMP (%d threads, schedule=%s) ...\n",
           omp_get_max_threads(), sched.c_str());
    fflush(stdout);

    /* ---- parallel compute ------------------------------------------------- */
    /* Apply schedule string to OpenMP runtime before the parallel region */
    if      (sched == "static")  omp_set_schedule(omp_sched_static,  0);
    else if (sched == "guided")  omp_set_schedule(omp_sched_guided,  0);
    else                         omp_set_schedule(omp_sched_dynamic, 16);

    double tc0 = omp_get_wtime();

    #pragma omp parallel for schedule(runtime)
    for (int id = 0; id < N; ++id) {
        const uint8_t *img = all_imgs.data() + (size_t)id * IMG_PX;

        int32_t best_score = INT32_MAX;
        int best_x = 0, best_y = 0;

        for (int y = 0; y <= IMG_H - TMPL_H; ++y) {
            for (int x = 0; x <= IMG_W - TMPL_W; ++x) {
                int32_t sc = ssd_at(img, tmpl, y, x);
                if (sc < best_score) {
                    best_score = sc;
                    best_x = x;
                    best_y = y;
                }
            }
        }

        res_x[id]     = best_x;
        res_y[id]     = best_y;
        res_score[id] = best_score;
    }

    double tc1 = omp_get_wtime();
    printf("Compute time : %.3f s\n", tc1 - tc0);

    /* ---- write results in order ------------------------------------------- */
    FILE *fo = fopen(out_path, "w");
    if (!fo) { perror(out_path); return 1; }
    for (int id = 0; id < N; ++id)
        fprintf(fo, "%d %d %d %d\n", id, res_x[id], res_y[id], res_score[id]);
    fclose(fo);

    printf("Done. Results -> %s\n", out_path);
    return 0;
}
