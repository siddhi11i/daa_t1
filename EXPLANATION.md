# EXPLANATION.md -- Parallel Template Matching: Full Explanation

---

## 1. The Problem in Simple Words

We have a large collection of grayscale images (each 256x256 pixels) and a small template
image (32x32 pixels).  The task: for each image, find the position where the template
matches best.

**"Matching" means:** place the template on top of the image at every possible position,
compute how different they are, and report the position with the smallest difference.

The difference metric is **SSD (Sum of Squared Differences):**

```
SSD(y, x) = sum over all template pixels (ty, tx) of:
                ( image[y+ty][x+tx] - template[ty][tx] )^2
```

A lower SSD means a better match.  SSD = 0 means a perfect, pixel-exact match.

### Worked example: 4x4 image, 2x2 template

```
Image (4x4):          Template (2x2):
  10  20  30  40         10  20
  50  60  70  80         50  60
  90 100 110 120
 130 140 150 160
```

The template can be placed at 3x3 = 9 positions: (y,x) from (0,0) to (2,2).

**Position (0,0):**
```
  SSD = (10-10)^2 + (20-20)^2 + (50-50)^2 + (60-60)^2
      = 0 + 0 + 0 + 0 = 0          <-- perfect match!
```

**Position (0,1):**
```
  SSD = (20-10)^2 + (30-20)^2 + (60-50)^2 + (70-60)^2
      = 100 + 100 + 100 + 100 = 400
```

**Position (1,0):**
```
  SSD = (50-10)^2 + (60-20)^2 + (100-50)^2 + (110-60)^2
      = 1600 + 1600 + 2500 + 2500 = 8200
```

The best position is (0,0) with SSD=0.

**Tie-break rule:** if two positions have the same SSD, the one with the smallest y wins;
if y is also equal, the one with the smallest x wins.  In our code this happens naturally
because we scan y from 0 upward and x from 0 rightward, and use strict less-than (`<`)
to update the best.

---

## 2. Why It Is Expensive, and Why It Suits Parallelism

### Cost formula

For one image of size H x W and a template of size h x w:

```
Positions to check  = (H - h + 1) x (W - w + 1)
Operations per pos  = h x w  (one subtraction, one multiply, one add per pixel)
Total per image     = (H-h+1) x (W-w+1) x h x w
```

With our values H=W=256, h=w=32:

```
Positions  = 225 x 225 = 50,625
Ops/pos    = 32 x 32   = 1,024
Total/img  = 50,625 x 1,024 = 51,840,000  (~52 million operations)
```

For N=5000 images: **259 billion operations.**  That is why the sequential version
takes ~25 seconds.

### Why it suits parallelism

- **Each image is independent.**  The result for image 0 does not depend on image 1.
  This is called "embarrassingly parallel" -- no synchronisation or communication
  is needed between workers during computation.
- **No shared mutable state.**  Each thread/rank reads from a shared read-only image
  buffer and writes to its own slot in the result array.
- **Even workload.**  Every image requires exactly the same amount of computation
  (50,625 x 1,024 ops), so static partitioning gives near-perfect load balance.

---

## 3. Dataset: How It Was Built

### Source images
`gen_data.py` reads 5000 JPEG images from the `val2017/` folder (COCO val2017 dataset),
converts each to grayscale, and resizes to 256x256.

### Template generation
A 32x32 "blocky" template is created from random 4x4 solid pixel blocks (8x8 blocks
total, each a random intensity 0-255).  The randomness is seeded with `SEED=42` so
the template is reproducible.  The blocky pattern ensures the template has high contrast
and is unlikely to occur naturally in any image.

### Planting the template
For ~30% of images (determined by `random.Random(SEED + N).random() < 0.30`), the
template is pasted into the image at a random position.  Because the bytes are
literally overwritten, the SSD at that position is exactly 0.

The ground truth CSV records which images were planted and at what (x, y):
```csv
image_id,planted,x,y
0,0,-1,-1
1,1,142,87
...
```

### Augmentation for N > 5000
We only have 5000 unique images.  To create larger datasets:
- Indices 5000-9999: horizontal flips of images 0-4999
- Indices 10000-49999: Gaussian noise (sigma=15) added to images 0-9999, cycling

This is valid because:
1. The matcher treats each image independently -- duplicates don't affect correctness.
2. Flips and noise change the pixel values, so the SSD computation is different for each.
3. The template is planted AFTER augmentation, so the planted patch is always exact.

### File format
- `images_N1000.bin`: raw bytes, 1000 x 256 x 256 = 65,536,000 bytes, no header.
- `template.bin`: 32 x 32 = 1024 bytes, no header.
- `gt_N1000.csv`: standard CSV with header row.

---

## 4. Sequential Code (seq.cpp), Function by Function

### `ssd_at(img, tmpl, y, x)` (lines 47-61)
Computes SSD at one window position.  Two nested loops over template rows (ty) and
columns (tx).  Returns a single `int32_t` score.  Declared `static inline` so the
compiler inlines it into the main loop for speed.

### `main()` (lines 63-148)
1. **Parse arguments:** `images.bin template.bin N output.txt`
2. **Load template** (1024 bytes from `template.bin`) and **all images** (N x 65536
   bytes into a `std::vector<uint8_t>`).
3. **Time the load** with `std::chrono::high_resolution_clock`.
4. **Compute loop** (lines 111-133): for each image id:
   - Set `best_score = INT32_MAX`, `best_x = best_y = 0`.
   - Loop y from 0 to 224, x from 0 to 224.
   - Call `ssd_at`.  If `sc < best_score`, update best.
   - The `<` (strict less-than) means the first position found wins ties -- which is
     (lowest y, lowest x) because that is the scan order.
5. **Time the compute** section separately.
6. **Write output:** one line per image: `image_id x y score`.

---

## 5. OpenMP Version (omp.cpp): What Changed

### The one-line change
```cpp
#pragma omp parallel for schedule(runtime)
for (int id = 0; id < N; ++id) {
```
This single pragma parallelises the outer loop over images.

### Shared vs private variables
- **Shared (read-only):** `all_imgs` (the image buffer), `tmpl` (the template),
  `res_x`, `res_y`, `res_score` (result arrays -- each thread writes to a different
  index, so no conflict).
- **Private (per-thread):** `id` (loop variable), `img` (pointer into all_imgs),
  `best_score`, `best_x`, `best_y`, `sc`, `y`, `x` -- all declared inside the loop
  body, so OpenMP automatically makes them private.

### Why there is no race condition
Each iteration writes only to `res_x[id]`, `res_y[id]`, `res_score[id]`, where `id`
is unique per iteration.  No two threads ever write to the same index.  The image
buffer and template are only read, never written.  Therefore no mutex, atomic, or
critical section is needed.

### Schedule selection
The schedule can be `static`, `dynamic`, or `guided`, set via `argv[6]` and applied
with `omp_set_schedule()` before the parallel region:
- **static:** each thread gets N/T consecutive images.  Lowest overhead.
- **dynamic(16):** threads grab chunks of 16 images from a shared work queue.  Better
  if some images take longer (not the case here, but safe default).
- **guided:** like dynamic but chunk size shrinks over time.

### Timing
Uses `omp_get_wtime()` (wall-clock time) around the parallel region only.

### Output determinism
Results are written AFTER the parallel region in sequential order (for id 0 to N-1),
so the output file is identical to seq.cpp regardless of thread scheduling.
---

## 6. MPI Version (mpi.cpp): How Work Is Split

### Architecture
MPI runs multiple independent **processes** (ranks), typically on different CPU cores
or even different machines.  Unlike OpenMP threads, MPI ranks do NOT share memory.

### Work distribution
1. **All ranks** know N from `argv[3]`.
2. Each rank computes its slice:
   ```
   chunk        = N / nproc
   rem          = N % nproc
   local_n      = chunk + (rank < rem ? 1 : 0)
   global_start = rank * chunk + min(rank, rem)
   ```
   Example: N=1000, 4 ranks -> each gets 250 images.

3. **MPI-IO:** each rank opens the same `images_N1000.bin` and reads only its own
   slice using `MPI_File_read_at(fh, global_start * 65536, ...)`.  Rank 0 does NOT
   need to hold the entire dataset in memory.

### Communication
- `MPI_Bcast`: rank 0 broadcasts the template (1024 bytes) to all ranks.
- `MPI_Barrier`: placed before and after the compute section so `MPI_Wtime` gives
  a fair wall-clock measurement.
- `MPI_Gatherv`: each rank sends its local_x, local_y, local_score arrays to rank 0.
  Three separate Gatherv calls (one per array).

### Rank 0 writes output
After gathering, rank 0 writes the output in order (id 0 to N-1), identical to seq.cpp.

---

## 7. Why We Chose These Parallelism Models

| Model | Strength | Best for |
|-------|----------|----------|
| **OpenMP** | Simple pragma-based threading, shared memory | Single multi-core machine |
| **MPI** | Distributed memory, works across machines | Clusters, HPC nodes |
| **Hybrid MPI+OpenMP** | MPI between nodes, OpenMP within each node | Large clusters |

Our problem is embarrassingly parallel (independent images), so all three models
work naturally.  OpenMP is the simplest to add (one `#pragma`).  MPI scales to
clusters.  We implemented both for comparison.

---

## 8. Verification and Edge Cases

### verify.py -- gt mode
For each planted image: asserts that the found (x,y) matches the ground-truth
exactly AND the score is 0.  For unplanted images: warns if score is 0 (potential
false positive).  Prints PASS only if all planted images are correctly detected.

### verify.py -- compare mode
Compares two output files line by line.  Prints PASS only if every line is identical.
Used to confirm omp and mpi produce the same output as seq.

### evaluate.py
Computes full metrics:
- **Correctly detected:** planted AND position matches exactly.
- **False positive:** unplanted but score = 0 (the matcher thinks a perfect match exists).
- **Correctly rejected:** unplanted and score > 0.
- **Precision, Recall, F1.**

### Edge cases handled
- Template at image boundary: the sliding window stops at `y = 224, x = 224`
  (= 256 - 32), ensuring the 32x32 window never goes out of bounds.
- Tie-break determinism: strict `<` with y-outer/x-inner scan order guarantees
  the same winner across all parallel versions.
- N not divisible by thread/rank count: the remainder `rem = N % nproc` is
  distributed one extra image to the first `rem` ranks.

---

## 9. Metrics: Time, Speedup, Efficiency

### Definitions

**Execution time (T):** wall-clock seconds for the compute section only (excluding
file I/O).

**Speedup (S):**
```
S(p) = T_sequential / T_parallel(p)
```
where p = number of threads or processes.  Ideal speedup = p.

**Efficiency (E):**
```
E(p) = S(p) / p = T_sequential / (p * T_parallel(p))
```
Ideal efficiency = 1.0 (100%).  Values below 1.0 indicate parallel overhead.

**Scaling:**
- **Strong scaling:** fix N, increase p.  Ideally T halves when p doubles.
- **Weak scaling:** increase N proportionally with p.  Ideally T stays constant.

### Our actual results (from runs on this machine)

| Version | Threads | N=1000 Time(s) | N=5000 Time(s) | Speedup (N=5000) | Efficiency |
|---------|---------|----------------|----------------|-------------------|------------|
| seq     | 1       | 5.192          | 25.548         | 1.00x             | 100.0%     |
| omp     | 1       | 5.221          | 25.614         | 1.00x             | 99.7%      |
| omp     | 2       | 3.662          | 13.056         | 1.96x             | 97.8%      |
| omp     | 4       | 1.740          | 8.875          | 2.88x             | 72.0%      |
| omp     | 8       | 1.022          | 4.847          | 5.27x             | 65.9%      |

### Classification & Accuracy Metrics (via evaluate.py)

| Metric | N=1000 | N=5000 | Description |
|--------|--------|--------|-------------|
| **Total Images** | 1000 | 5000 | Total evaluated test images |
| **Planted Targets** | 300 (30.0%) | 1498 (30.0%) | Ground-truth planted templates |
| **Correctly Detected** | 300 (100%) | 1498 (100%) | Correct (x, y) location with score = 0 |
| **Wrong Position** | 0 | 0 | Detection at wrong coordinates |
| **False Positives** | 0 | 0 | Score = 0 on non-planted images |
| **Precision** | **1.0000** | **1.0000** | TP / (TP + FP) |
| **Recall** | **1.0000** | **1.0000** | TP / (TP + FN) |
| **F1 Score** | **1.0000** | **1.0000** | Harmonic mean of precision & recall |

### How to decide which version is better
1. **Correctness first:** all versions must produce identical output (verify.py compare) and 100% F1 score.
2. **Speedup:** omp-8 gives 5.27x speedup on N=5000 -- the fastest single-machine option (runtime reduced from ~25.5s down to 4.85s).
3. **Efficiency:** omp-2 achieves near-perfect 97.8% efficiency, while omp-4 maintains 72.0% and omp-8 achieves 65.9%.
4. **For distributed systems:** MPI can scale horizontally beyond one machine's socket/core limits across a cluster.
5. **Rule of thumb:** use the smallest thread count that meets your wall-clock latency deadline while preserving high efficiency.

---

## 10. Bottlenecks Seen in Our Results

1. **Sub-linear speedup (5.28x with 8 threads instead of 8x):**
   The main bottleneck is **memory bandwidth**.  Each thread reads a different 256x256
   image and the 32x32 template from RAM.  With 8 threads all competing for the same
   L3 cache and DRAM bus, the CPU cores spend time waiting for data.

2. **Load time is negligible:**
   Loading 5000 images (312 MB) takes 0.19s vs 25.5s compute.  I/O is not a bottleneck.

3. **Linear scaling with N:**
   5.19s for N=1000 vs 25.55s for N=5000 (ratio = 4.92 ~ 5x).  This confirms O(N)
   complexity and shows no overhead growth with dataset size.

4. **Dynamic schedule overhead is minimal:**
   With 50,625 identical-cost positions per image, the work is perfectly balanced.
   Dynamic scheduling adds a tiny overhead for the work-queue but provides safety
   in case of OS-level jitter.

---

## 11. Alternatives and Improvements

| Technique | Benefit | Trade-off |
|-----------|---------|-----------|
| **NCC (Normalised Cross-Correlation)** | Invariant to brightness/contrast changes | More expensive per position (needs mean and stddev); our planted template is exact so SSD suffices |
| **Early termination** | Skip remaining pixels once partial SSD exceeds current best | Adds a branch per pixel; benefits depend on data; worst case no improvement |
| **Integral images (SAT)** | Precompute prefix sums to get SSD in O(1) per position | Requires precomputing sum-of-squares tables; only helps for repeated templates on the same image |
| **GPU shared-memory tiling (CUDA)** | 1000x+ speedup; template fits in constant memory (1 KB) | Requires NVIDIA GPU; data transfer to/from device; more complex code |
| **FFT-based matching** | Reduces per-image cost from O(H*W*h*w) to O(H*W*log(H*W)) | FFT has high constant overhead; only wins for large templates (h,w > ~64) |
| **SIMD (SSE/AVX)** | 4-8x speedup of the inner loop via vectorised subtract-and-square | Compiler may auto-vectorise with -O3; manual SIMD is platform-specific |
| **Hybrid MPI + OpenMP** | MPI between nodes, OpenMP within each node for best of both | More complex setup; overkill for single-machine use |
| **Batching for large N** | Process images in chunks to limit peak memory usage | Adds I/O passes; our memmap approach in gen_data.py already handles this |
| **Multi-scale search** | Build image pyramid, match at coarse scale first, refine | Misses small templates; our 32x32 template is already small |
| **Hashing / feature-based** | Skip most positions using keypoint descriptors (SIFT, ORB) | Not suitable for exact pixel-level SSD matching; changes the problem definition |

---

## 12. Viva Questions and Answers

**Q1: What is SSD and why did you choose it over other metrics?**
SSD (Sum of Squared Differences) sums the squared pixel-wise differences between the
template and the image patch.  We chose it because (a) it is simple and fast to compute,
(b) it uses only integer arithmetic (no division), and (c) our planted template is
an exact byte copy, so SSD = 0 is guaranteed for correct matches.

**Q2: What is the time complexity of the brute-force approach?**
O(N * (H-h+1) * (W-w+1) * h * w).  With our values: O(N * 225 * 225 * 32 * 32)
= O(N * 51,840,000).  This is O(N) since the per-image cost is constant.

**Q3: How does the tie-break rule work?**
We use strict less-than (`if (sc < best_score)`) and scan y from 0 upward, x from 0
rightward.  The first position encountered wins, which is always the one with the
smallest y, then smallest x.

**Q4: What is the difference between OpenMP and MPI?**
OpenMP uses shared memory -- threads in the same process access the same RAM.
MPI uses message passing -- processes have separate memory and communicate by
sending/receiving data.  OpenMP is simpler but limited to one machine.
MPI works across networked machines.

**Q5: Why is there no race condition in omp.cpp?**
Each thread writes to `res_x[id]`, `res_y[id]`, `res_score[id]` where `id` is the
loop iteration variable (unique per thread).  The image buffer and template are
read-only.  No two threads access the same memory location for writing.

**Q6: What does `schedule(runtime)` mean?**
It tells OpenMP to read the schedule type from the `OMP_SCHEDULE` environment variable
or from `omp_set_schedule()` at runtime.  This lets us switch between static, dynamic,
and guided without recompiling.

**Q7: How does MPI-IO differ from rank-0-reads-and-scatters?**
With MPI-IO (`MPI_File_read_at`), each rank reads its own slice directly from disk.
This avoids rank 0 needing enough RAM to hold the entire dataset and avoids the
scatter communication overhead.

**Q8: Why is speedup sub-linear (e.g. 5.28x with 8 threads)?**
Memory bandwidth saturation.  All 8 threads compete for the L3 cache and DRAM bus.
The inner loop reads 32x32 = 1024 bytes per position, and there are 50,625 positions
per image.  OS scheduling and thread creation overhead also contribute.

**Q9: What is parallel efficiency and what does 66% mean?**
Efficiency = Speedup / Threads.  66% means each thread contributes only 66% of its
potential speedup.  The remaining 34% is lost to overhead (memory contention, OS
scheduling).  Above 50% is generally considered good for shared-memory parallelism.

**Q10: How did you verify correctness of the parallel versions?**
Two methods: (1) `verify.py gt` checks each planted image's position and score against
ground truth.  (2) `verify.py compare` does a line-by-line diff between sequential and
parallel output files.  Both must print PASS.

**Q11: What would change if the template were larger (e.g. 128x128)?**
The cost per position would increase from 1024 to 16384 ops.  The number of positions
would decrease from 225x225 to 129x129.  Overall cost per image would increase from
52M to 216M ops.  FFT-based matching might become more efficient for such large templates.

**Q12: Why do you load all images into memory before computing?**
To avoid I/O during the timed compute section, and to let OpenMP threads access any
image by index without file seeking.  For N=5000 this requires ~312 MB, which fits
comfortably in modern RAM.

**Q13: Could you parallelise the inner loop (over positions) instead?**
Yes, but it would require a parallel reduction to find the minimum SSD.  The inner
loop has 50,625 iterations, each very cheap (1024 ops), so the overhead of spawning
threads or synchronising a reduction would likely outweigh the benefit.  Parallelising
over images (the outer loop) is more efficient.

**Q14: What is the purpose of MPI_Barrier before timing?**
It ensures all ranks have finished loading data before the timer starts.  Without it,
a slow rank's load time would be counted as part of a fast rank's compute time.

**Q15: How does your planted template guarantee SSD = 0?**
`plant_tmpl(img, template, px, py)` copies the template's exact bytes into the image
at position (px, py).  Since SSD computes (image_pixel - template_pixel)^2, and every
pixel is identical after planting, every term is 0, so SSD = 0.

**Q16: What happens if two different positions both have SSD = 0?**
The tie-break rule picks the one with the smallest y, then smallest x.  In practice
this cannot happen with our blocky random template -- it is too distinctive to match
any natural image region by accident.

**Q17: Why use `int32_t` for SSD instead of `int`?**
Each pixel difference can be at most 255.  Squared: 65025.  Summed over 1024 pixels:
max SSD = 1024 * 65025 = 66,585,600.  This fits in int32 (max ~2.1 billion) with
room to spare.  Using int32_t makes the size explicit and portable.

**Q18: What is `np.memmap` and why use it in gen_data.py?**
`np.memmap` creates a memory-mapped file -- it writes data directly to disk without
holding the entire array in RAM.  For N=50000, the output is 3.2 GB; memmap avoids
needing 3.2 GB of free RAM.

**Q19: How would you implement this on a GPU (CUDA)?**
Launch one CUDA thread per (y, x) position (225x225 = 50,625 threads per image).
Store the template in constant memory (1 KB, broadcast to all threads).  Each thread
computes SSD for its position.  Use a block-level parallel reduction to find the
minimum SSD within each thread block, then a grid-level reduction for the global
minimum per image.

**Q20: What is the difference between strong and weak scaling?**
Strong scaling: fix N, increase processors.  Measures how much faster you can solve
the SAME problem.  Weak scaling: increase N proportionally with processors.  Measures
how large a problem you can solve in the SAME time.  Our benchmark.sh tests strong
scaling (fixed N, varying threads/ranks).
