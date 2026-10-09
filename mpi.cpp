/*
 * mpi.cpp  --  Step 5: MPI parallel brute-force SSD template matching
 *
 * Parallelism strategy
 * --------------------
 *   Each rank reads ONLY its own slice of images_N.bin via byte offsets
 *   (MPI_File_read_at), avoiding the need for rank 0 to hold the full dataset.
 *   The template is read by rank 0 and broadcast.
 *   Local matches are computed serially per rank (or with optional OpenMP
 *   if compiled with -fopenmp; controlled by OMP_NUM_THREADS).
 *   Results are gathered with MPI_Gatherv to rank 0, which writes the output.
 *
 * Build
 * -----
 *   mpicxx -O3 -o mpi mpi.cpp
 *   mpicxx -O3 -fopenmp -o mpi mpi.cpp   # hybrid version
 *
 * Usage
 * -----
 *   mpirun -np 4 ./mpi <images.bin> <template.bin> <N> <output.txt>
 *
 * Timing
 * ------
 *   MPI_Barrier around the compute section; MPI_Wtime for timing.
 *   Rank 0 prints load time and compute time.
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
#include <mpi.h>

static const int IMG_H   = 256;
static const int IMG_W   = 256;
static const int TMPL_H  = 32;
static const int TMPL_W  = 32;
static const int IMG_PX  = IMG_H * IMG_W;    // 65536
static const int TMPL_PX = TMPL_H * TMPL_W; // 1024

/* Compute SSD at window position (y, x). */
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
    MPI_Init(&argc, &argv);
    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    if (argc < 5) {
        if (rank == 0)
            fprintf(stderr,
                "Usage: mpirun -np P %s <images.bin> <template.bin> <N> <output.txt>\n",
                argv[0]);
        MPI_Finalize();
        return 1;
    }

    const char *img_path  = argv[1];
    const char *tmpl_path = argv[2];
    int         N         = atoi(argv[3]);
    const char *out_path  = argv[4];

    /* ---- broadcast N (it is already known from argv, but keep consistent) -- */
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* ---- load template: rank 0 reads, broadcasts to all ------------------- */
    double t_load0 = MPI_Wtime();

    uint8_t tmpl[TMPL_PX];
    if (rank == 0) {
        FILE *ft = fopen(tmpl_path, "rb");
        if (!ft) { perror(tmpl_path); MPI_Abort(MPI_COMM_WORLD, 1); }
        if (fread(tmpl, 1, TMPL_PX, ft) != (size_t)TMPL_PX) {
            fprintf(stderr, "Template too small\n"); MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fclose(ft);
    }
    MPI_Bcast(tmpl, TMPL_PX, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    /* ---- compute each rank's slice of images ------------------------------ */
    int chunk        = N / nproc;
    int rem          = N % nproc;
    int local_n      = chunk + (rank < rem ? 1 : 0);
    int global_start = rank * chunk + (rank < rem ? rank : rem);

    /* Each rank reads its own slice directly from disk via MPI-IO */
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, img_path,
                  MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    MPI_Offset byte_offset = (MPI_Offset)global_start * IMG_PX;
    std::vector<uint8_t> local_imgs((size_t)local_n * IMG_PX);
    MPI_File_read_at(fh, byte_offset,
                     local_imgs.data(), local_n * IMG_PX,
                     MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);

    double t_load1 = MPI_Wtime();
    if (rank == 0)
        printf("Load time    : %.3f s\n", t_load1 - t_load0);

    /* ---- barrier before compute timing ------------------------------------ */
    MPI_Barrier(MPI_COMM_WORLD);
    double tc0 = MPI_Wtime();

    /* ---- local matching --------------------------------------------------- */
    std::vector<int>     local_x(local_n), local_y(local_n);
    std::vector<int32_t> local_score(local_n);

    for (int i = 0; i < local_n; ++i) {
        const uint8_t *img = local_imgs.data() + (size_t)i * IMG_PX;
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

        local_x[i]     = best_x;
        local_y[i]     = best_y;
        local_score[i] = best_score;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double tc1 = MPI_Wtime();
    if (rank == 0)
        printf("Compute time : %.3f s\n", tc1 - tc0);

    /* ---- gather results to rank 0 ---------------------------------------- */
    std::vector<int> gcounts(nproc), gdispls(nproc);
    {
        int off = 0;
        for (int r = 0; r < nproc; ++r) {
            gcounts[r] = chunk + (r < rem ? 1 : 0);
            gdispls[r] = off;
            off += gcounts[r];
        }
    }

    std::vector<int>     all_x, all_y;
    std::vector<int32_t> all_score;
    if (rank == 0) {
        all_x.resize(N); all_y.resize(N); all_score.resize(N);
    }

    MPI_Gatherv(local_x.data(), local_n, MPI_INT,
                all_x.data(), gcounts.data(), gdispls.data(), MPI_INT,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(local_y.data(), local_n, MPI_INT,
                all_y.data(), gcounts.data(), gdispls.data(), MPI_INT,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(local_score.data(), local_n, MPI_INT,
                all_score.data(), gcounts.data(), gdispls.data(), MPI_INT,
                0, MPI_COMM_WORLD);

    /* ---- rank 0 writes output --------------------------------------------- */
    if (rank == 0) {
        FILE *fo = fopen(out_path, "w");
        if (!fo) { perror(out_path); MPI_Finalize(); return 1; }
        for (int id = 0; id < N; ++id)
            fprintf(fo, "%d %d %d %d\n", id, all_x[id], all_y[id], all_score[id]);
        fclose(fo);
        printf("Done. Results -> %s\n", out_path);
    }

    MPI_Finalize();
    return 0;
}
