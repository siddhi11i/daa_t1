/*
 * seq.cpp  --  Step 2: Sequential brute-force SSD template matching
 *
 * Algorithm
 * ---------
 *   For each image in the binary pack:
 *     Slide a 32x32 window over the 256x256 image.
 *     Compute SSD (sum of squared differences, int32) at every position.
 *     Report the position with the smallest SSD.
 *     Tie-break: smallest y first, then smallest x.
 *
 * Build
 * -----
 *   g++ -O3 -o seq seq.cpp
 *
 * Usage
 * -----
 *   ./seq <images.bin> <template.bin> <N> <output.txt>
 *
 * Timing
 * ------
 *   File-loading time and compute time are printed separately.
 *   Only the compute section is timed for the benchmark.
 *
 * Output format (one line per image)
 * ------------------------------------
 *   image_id x y score
 */

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>
#include <cassert>
#include <chrono>

static const int IMG_H  = 256;
static const int IMG_W  = 256;
static const int TMPL_H = 32;
static const int TMPL_W = 32;
static const int IMG_PX   = IMG_H * IMG_W;     // 65536
static const int TMPL_PX  = TMPL_H * TMPL_W;  // 1024

/* Compute SSD at a single window position (y, x) inside img. */
static inline int32_t ssd_at(const uint8_t *img,
                              const uint8_t *tmpl,
                              int y, int x)
{
    int32_t s = 0;
    for (int ty = 0; ty < TMPL_H; ++ty) {
        const uint8_t *row_i = img  + (y + ty) * IMG_W + x;
        const uint8_t *row_t = tmpl + ty * TMPL_W;
        for (int tx = 0; tx < TMPL_W; ++tx) {
            int d = (int)row_i[tx] - (int)row_t[tx];
            s += d * d;
        }
    }
    return s;
}

int main(int argc, char *argv[])
{
    if (argc < 5) {
        fprintf(stderr,
            "Usage: %s <images.bin> <template.bin> <N> <output.txt>\n",
            argv[0]);
        return 1;
    }

    const char *img_path  = argv[1];
    const char *tmpl_path = argv[2];
    int         N         = atoi(argv[3]);
    const char *out_path  = argv[4];

    using Clock = std::chrono::high_resolution_clock;
    using Sec   = std::chrono::duration<double>;

    /* ---- load template ----------------------------------------------------- */
    auto t0 = Clock::now();

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

    auto t1 = Clock::now();
    double load_sec = Sec(t1 - t0).count();
    printf("Load time : %.3f s\n", load_sec);

    /* ---- compute: sliding-window SSD for every image ---------------------- */
    auto tc0 = Clock::now();

    std::vector<int>     res_x(N), res_y(N);
    std::vector<int32_t> res_score(N);

    for (int id = 0; id < N; ++id) {
        const uint8_t *img = all_imgs.data() + (size_t)id * IMG_PX;

        int32_t best_score = INT32_MAX;
        int best_x = 0, best_y = 0;

        /* Slide window; tie-break: lowest y first, then lowest x */
        for (int y = 0; y <= IMG_H - TMPL_H; ++y) {
            for (int x = 0; x <= IMG_W - TMPL_W; ++x) {
                int32_t sc = ssd_at(img, tmpl, y, x);
                /* strict less-than => first (lowest-y, lowest-x) wins */
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

    auto tc1 = Clock::now();
    double compute_sec = Sec(tc1 - tc0).count();
    printf("Compute time : %.3f s\n", compute_sec);

    /* ---- write output ----------------------------------------------------- */
    FILE *fo = fopen(out_path, "w");
    if (!fo) { perror(out_path); return 1; }
    for (int id = 0; id < N; ++id)
        fprintf(fo, "%d %d %d %d\n", id, res_x[id], res_y[id], res_score[id]);
    fclose(fo);

    printf("Done. Results -> %s\n", out_path);
    return 0;
}
