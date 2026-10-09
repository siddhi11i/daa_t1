# RUN.md -- Execution Guide for Parallel Template Matching

This guide details how to build, run, verify, and benchmark the Parallel Template Matching project across Sequential, OpenMP, and MPI implementations.

---

## 1. Prerequisites

Ensure the following tools and libraries are installed on your Linux / WSL / MSYS2 environment:

| Component | Minimum Version | Installation Command (Ubuntu/Debian) | Verification Command |
|---|---|---|---|
| **g++** | 9+ (C++17 support) | `sudo apt install g++` | `g++ --version` |
| **OpenMP** | Included with GCC | `sudo apt install libgomp1` | `echo | g++ -fopenmp -x c++ -` |
| **MPI Compiler & Runtime** | OpenMPI 4+ or MPICH | `sudo apt install libopenmpi-dev openmpi-bin` | `mpicxx --version && mpiexec --version` |
| **CUDA Toolkit** *(optional)* | CUDA 11+ | Follow NVIDIA CUDA installation | `nvcc --version` |
| **Python** | 3.8+ | `sudo apt install python3 python3-pip` | `python3 --version` |
| **Python Packages** | NumPy, Pillow, Pandas, Matplotlib | `pip install numpy Pillow pandas matplotlib` | `python3 -c "import numpy, PIL, pandas, matplotlib; print('OK')"` |
| **COCO val2017 Dataset** | 5,000 JPEG images | Download COCO val2017 into `val2017/` | `ls -1 val2017/*.jpg \| wc -l` (must be ≥ 5000) |

---

## 2. Folder Structure

```
DAA_T1/
├── val2017/                  # 5,000 COCO val2017 raw grayscale/color JPEG images
├── template.bin              # 32x32 binary template (1,024 bytes, uint8 row-major)
├── images_N1000.bin          # Binary image dataset N=1,000 (65,536,000 bytes)
├── images_N5000.bin          # Binary image dataset N=5,000 (327,680,000 bytes)
├── gt_N1000.csv              # Ground truth metadata for N=1,000 (image_id, planted, x, y)
├── gt_N5000.csv              # Ground truth metadata for N=5,000 (image_id, planted, x, y)
├── gen_data.py               # Dataset generator script using NumPy memmap
├── seq.cpp                   # Sequential C++ brute-force SSD template matcher
├── omp.cpp                   # OpenMP multi-threaded C++ template matcher
├── mpi.cpp                   # MPI distributed-memory C++ template matcher (MPI-IO)
├── verify.py                 # Ground-truth validator & output comparison script
├── evaluate.py               # Precision, recall, and F1 score computation script
├── benchmark.sh              # 5-repetition automated benchmarking shell script
├── results.csv               # Recorded benchmark execution timings
├── plot.py                   # Matplotlib performance graphing script
├── plot_*.png                # Generated performance plots (time, speedup, efficiency, scaling)
├── Makefile                  # Build automation for all C++ binaries
├── run_all.sh                # End-to-end automated pipeline script (stops on error)
├── RUN.md                    # This step-by-step execution documentation
├── EXPLANATION.md            # Comprehensive project and algorithm explanation
└── llm_log.md                # Development and debugging history log
```

---

## 3. End-to-End Automated Run

To execute the entire project end-to-end (dataset generation, compilation, execution, verification, benchmarking, and graph plotting) with automatic failure detection:

```bash
chmod +x run_all.sh
./run_all.sh
```

`run_all.sh` executes with `set -e` and stops immediately upon any error, displaying `[PASS]` or `[FAIL]` at every stage.

---

## 4. Step-by-Step Commands (One by One)

Each command below corresponds directly to a stage in the pipeline:

### Stage 1: Generate Dataset
```bash
python3 gen_data.py
```
*Generates the 32x32 synthetic template, resizes 5,000 COCO images to 256x256, plants the template in ~30% of images, and exports raw binary files and ground-truth CSVs.*

### Stage 2: Compile C++ Matchers with -O3
```bash
g++ -O3 -Wall -o seq seq.cpp
```
*Compiles the sequential template matcher with maximum compiler optimization (-O3).*

```bash
g++ -O3 -Wall -fopenmp -o omp omp.cpp
```
*Compiles the OpenMP multi-threaded template matcher with OpenMP runtime support.*

```bash
mpicxx -O3 -Wall -o mpi mpi.cpp
```
*Compiles the MPI distributed-memory template matcher using the MPI C++ compiler wrapper.*

*(Optional CUDA build, only if `nvcc` and `cuda.cu` exist):*
```bash
nvcc -O3 -o cuda cuda.cu
```
*Compiles the GPU-accelerated CUDA kernel with NVIDIA compiler.*

*(Alternatively, compile all C++ binaries at once using the Makefile):*
```bash
make all
```
*Builds `seq`, `omp`, and `mpi` binaries using the predefined Makefile rules.*

### Stage 3: Run Sequential Matcher for Each N
```bash
./seq images_N1000.bin template.bin 1000 out_seq_N1000.txt
```
*Executes sequential template matching on 1,000 images and writes coordinates and SSD scores.*

```bash
./seq images_N5000.bin template.bin 5000 out_seq_N5000.txt
```
*Executes sequential template matching on 5,000 images and writes coordinates and SSD scores.*

### Stage 4: Verify Sequential Output Against Ground Truth
```bash
python3 verify.py gt gt_N1000.csv out_seq_N1000.txt
```
*Verifies that all planted images in N=1,000 are detected at the exact (x, y) location with SSD = 0.*

```bash
python3 evaluate.py out_seq_N1000.txt gt_N1000.csv
```
*Calculates precision, recall, and F1 score against the ground truth for N=1,000.*

```bash
python3 verify.py gt gt_N5000.csv out_seq_N5000.txt
```
*Verifies that all planted images in N=5,000 are detected at the exact (x, y) location with SSD = 0.*

```bash
python3 evaluate.py out_seq_N5000.txt gt_N5000.csv
```
*Calculates precision, recall, and F1 score against the ground truth for N=5,000.*

### Stage 5: Run OpenMP and MPI Parallel Matchers
```bash
./omp images_N1000.bin template.bin 1000 out_omp4_N1000.txt 4 dynamic
```
*Runs the OpenMP matcher on 1,000 images with 4 threads using dynamic scheduling.*

```bash
./omp images_N1000.bin template.bin 1000 out_omp8_N1000.txt 8 dynamic
```
*Runs the OpenMP matcher on 1,000 images with 8 threads using dynamic scheduling.*

```bash
./omp images_N5000.bin template.bin 5000 out_omp4_N5000.txt 4 dynamic
```
*Runs the OpenMP matcher on 5,000 images with 4 threads using dynamic scheduling.*

```bash
./omp images_N5000.bin template.bin 5000 out_omp8_N5000.txt 8 dynamic
```
*Runs the OpenMP matcher on 5,000 images with 8 threads using dynamic scheduling.*

```bash
mpirun --oversubscribe -np 4 ./mpi images_N1000.bin template.bin 1000 out_mpi4_N1000.txt
```
*Runs the MPI matcher across 4 processes using MPI-IO parallel file reads.*

```bash
mpirun --oversubscribe -np 4 ./mpi images_N5000.bin template.bin 5000 out_mpi4_N5000.txt
```
*Runs the MPI matcher across 4 processes on 5,000 images.*

### Stage 6: Compare Parallel Outputs with Sequential Output
```bash
python3 verify.py compare out_seq_N1000.txt out_omp4_N1000.txt
```
*Performs a strict line-by-line comparison verifying OpenMP output is identical to sequential output.*

```bash
python3 verify.py compare out_seq_N1000.txt out_mpi4_N1000.txt
```
*Performs a strict line-by-line comparison verifying MPI output is identical to sequential output.*

### Stage 7: Run Benchmark Suite
```bash
chmod +x benchmark.sh
./benchmark.sh
```
*Runs sequential, OpenMP (1, 2, 4, 8 threads), and MPI (1, 2, 4, 8 ranks) 5 times each and logs timings to results.csv.*

### Stage 8: Generate Performance Graphs
```bash
python3 plot.py
```
*Parses results.csv and renders 4 analytical charts: execution time, speedup curve, parallel efficiency, and scaling vs N.*

---

## 5. How to Change N and Thread Counts

### In `run_all.sh`
Modify the configuration header at the top of [`run_all.sh`](file:///d:/DAA_T1/run_all.sh):
```bash
SIZES="1000 5000 10000 50000"     # Add or remove N values
THREADS="1 2 4 8 16"              # Adjust OpenMP thread counts
MPI_PROCS="1 2 4 8"               # Adjust MPI process counts
SCHEDULE="static"                 # Change scheduling: static | dynamic | guided
```

### In `gen_data.py`
To generate larger or custom dataset sizes, adjust the `SIZES` list in [`gen_data.py`](file:///d:/DAA_T1/gen_data.py):
```python
SIZES = [1000, 5000, 10000, 50000]
```

### In `benchmark.sh`
Update loop variables in [`benchmark.sh`](file:///d:/DAA_T1/benchmark.sh):
```bash
REPS=5                                    # Number of test repetitions
for N in 1000 5000; do                   # Target dataset sizes
for T in 1 2 4 8; do                     # OpenMP threads
for P in 1 2 4 8; do                     # MPI processes
```

### Direct Command-Line Invocations
- **Change OpenMP threads & schedule:**
  ```bash
  ./omp images_N1000.bin template.bin 1000 out_omp.txt <num_threads> <static|dynamic|guided>
  ```
- **Change MPI process count:**
  ```bash
  mpirun -np <num_procs> ./mpi images_N1000.bin template.bin 1000 out_mpi.txt
  ```

---

## 6. Common Errors and Fixes

| Error Message | Probable Cause | Fix |
|---|---|---|
| `g++: command not found` | GCC compiler is not installed | Run `sudo apt update && sudo apt install g++` (or install GCC via MSYS2). |
| `mpicxx: command not found` or `mpirun: command not found` | OpenMPI / MPICH is not installed | Run `sudo apt install libopenmpi-dev openmpi-bin`. |
| `fatal error: omp.h: No such file or directory` | OpenMP library header missing | Ensure `g++` is installed with `libgomp` (`sudo apt install libgomp1`). |
| `AssertionError: len(jpgs) >= 5000` | Missing COCO val2017 dataset images | Download full COCO val2017 dataset and extract at least 5,000 `.jpg` files into `val2017/`. |
| `Template file too small` or `Cannot open template.bin` | Template binary missing or ungenerated | Run `python3 gen_data.py` to create `template.bin`. |
| `Failed to read all images` | Binary file size does not match requested $N$ | Verify file size: `ls -lh images_N*.bin`. For $N=1000$, file must be exactly $1000 \times 256 \times 256 = 65,536,000$ bytes. |
| `FAIL: line count differs` in `verify.py compare` | A process crashed or failed before finishing all $N$ images | Check command output logs or rerun with fewer processes to inspect errors. |
| `WRONG img <id>: expected (x,y) got (px,py)` in `verify.py gt` | Tie-breaking logic mismatch or incorrect sliding window bounds | Ensure window loop stops at $256 - 32 = 224$ and strict `<` is used for score updates. |
| `There are not enough slots available in the system` (MPI) | `mpirun` refuses to launch more ranks than physical cores | Pass `--oversubscribe` flag: `mpirun --oversubscribe -np 8 ./mpi ...`. |
| `ModuleNotFoundError: No module named 'PIL'` | Python Pillow package is missing | Run `pip install Pillow`. |
| `ModuleNotFoundError: No module named 'matplotlib'` | Matplotlib / Pandas missing | Run `pip install pandas matplotlib`. |
| `collect2.exe: fatal error: CreateProcess` (Windows native) | Broken MinGW/Cygwin linker path | Use Linux/WSL or standard MSYS2 UCRT64 toolchain environment. |
