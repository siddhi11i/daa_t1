#!/usr/bin/env bash
# ==============================================================================
# run_all.sh -- End-to-End Execution Script for Parallel Template Matching
#
# Runs the whole project pipeline end to end and stops on the first error (set -e).
# Prints an echo banner before each stage, PASS/FAIL for every check,
# and a final summary.
#
# Stages:
#   1. Generate dataset
#   2. Compile seq/omp/mpi with -O3 (cuda only if nvcc exists)
#   3. Run sequential for each N
#   4. Verify against ground truth
#   5. Run omp and mpi
#   6. Compare each output with the sequential one
#   7. Run benchmark
#   8. Generate graphs
#
# Usage:
#   chmod +x run_all.sh && ./run_all.sh
# ==============================================================================

set -e
set -o pipefail

# ==============================================================================
# Configurable Variables & Paths
# ==============================================================================
SIZES="1000 5000"                    # List of N values to test (e.g. "1000 5000 10000 50000")
THREADS="1 2 4 8"                    # OpenMP thread counts
MPI_PROCS="1 2 4 8"                  # MPI process/rank counts
SCHEDULE="dynamic"                   # OpenMP schedule: static | dynamic | guided

# File and Directory Paths
SRC_DIR="val2017"                    # Directory containing COCO val2017 JPEG images
TEMPLATE_FILE="template.bin"         # 32x32 binary template
RESULTS_CSV="results.csv"            # Benchmark timing log
BENCHMARK_SCRIPT="benchmark.sh"      # Benchmark runner script

# Executables and Compilers
CXX="g++"
MPICXX="mpicxx"
NVCC="nvcc"
PYTHON="python3"

# Fallback to 'python' if 'python3' is not in PATH
if ! command -v "$PYTHON" &>/dev/null && command -v python &>/dev/null; then
    PYTHON="python"
fi

pass_count=0
fail_count=0

# Helper: run command, print [PASS] or [FAIL], and stop on first error
check() {
    local label="$1"
    shift
    if "$@"; then
        echo "  [PASS] $label"
        pass_count=$((pass_count + 1))
    else
        echo "  [FAIL] $label"
        fail_count=$((fail_count + 1))
        echo "ERROR: Step '$label' failed! Aborting pipeline."
        exit 1
    fi
}

echo "======================================================================"
echo "    PARALLEL TEMPLATE MATCHING: END-TO-END PIPELINE RUNNER"
echo "======================================================================"
echo "Dataset sizes (N)   : $SIZES"
echo "OpenMP thread counts: $THREADS"
echo "MPI process counts  : $MPI_PROCS"
echo "OpenMP schedule     : $SCHEDULE"
echo "Python interpreter  : $PYTHON"
echo "======================================================================"

# ==============================================================================
# Stage 1: Generate dataset
# ==============================================================================
echo ""
echo "======================================================================"
echo "Stage 1: Generate dataset"
echo "======================================================================"
data_missing=0
for N in $SIZES; do
    if [ ! -f "images_N${N}.bin" ] || [ ! -f "gt_N${N}.csv" ]; then
        data_missing=1
    fi
done
if [ ! -f "$TEMPLATE_FILE" ]; then
    data_missing=1
fi

if [ "$data_missing" -eq 1 ]; then
    echo "Dataset or template missing. Running gen_data.py..."
    check "Dataset generation (gen_data.py)" $PYTHON gen_data.py
else
    echo "All dataset files and template already present. Skipping gen_data.py."
    echo "  [PASS] Dataset verification (files exist)"
    pass_count=$((pass_count + 1))
fi

# ==============================================================================
# Stage 2: Compile seq/omp/mpi with -O3 (cuda only if nvcc exists)
# ==============================================================================
echo ""
echo "======================================================================"
echo "Stage 2: Compile seq, omp, mpi with -O3 (cuda only if nvcc exists)"
echo "======================================================================"

echo "Compiling sequential (seq.cpp) with -O3..."
check "Compile seq (g++ -O3)" $CXX -O3 -Wall -o seq seq.cpp

echo "Compiling OpenMP (omp.cpp) with -O3 -fopenmp..."
check "Compile omp (g++ -O3 -fopenmp)" $CXX -O3 -Wall -fopenmp -o omp omp.cpp

if command -v "$MPICXX" &>/dev/null; then
    echo "Compiling MPI (mpi.cpp) with -O3..."
    check "Compile mpi (mpicxx -O3)" $MPICXX -O3 -Wall -o mpi mpi.cpp
    HAS_MPI=1
else
    echo "Notice: mpicxx not found in PATH. Skipping MPI compilation."
    HAS_MPI=0
fi

if command -v "$NVCC" &>/dev/null && [ -f "cuda.cu" ]; then
    echo "Compiling CUDA (cuda.cu) with -O3..."
    check "Compile cuda (nvcc -O3)" $NVCC -O3 -o cuda cuda.cu
    HAS_CUDA=1
else
    echo "Notice: nvcc not found or cuda.cu not present. Skipping CUDA compilation."
    HAS_CUDA=0
fi

# ==============================================================================
# Stage 3: Run sequential for each N
# ==============================================================================
echo ""
echo "======================================================================"
echo "Stage 3: Run sequential for each N"
echo "======================================================================"
for N in $SIZES; do
    BIN="images_N${N}.bin"
    OUT="out_seq_N${N}.txt"
    echo "Running sequential matcher for N=$N..."
    check "Run seq N=$N" ./seq "$BIN" "$TEMPLATE_FILE" "$N" "$OUT"
done

# ==============================================================================
# Stage 4: Verify against ground truth
# ==============================================================================
echo ""
echo "======================================================================"
echo "Stage 4: Verify sequential against ground truth"
echo "======================================================================"
for N in $SIZES; do
    GT="gt_N${N}.csv"
    OUT="out_seq_N${N}.txt"
    echo "Verifying sequential output against ground truth for N=$N..."
    check "Verify seq N=$N vs GT" $PYTHON verify.py gt "$GT" "$OUT"
    echo "Evaluating precision, recall, and F1 score for N=$N..."
    check "Evaluate seq N=$N metrics" $PYTHON evaluate.py "$OUT" "$GT"
done

# ==============================================================================
# Stage 5: Run omp and mpi
# ==============================================================================
echo ""
echo "======================================================================"
echo "Stage 5: Run omp and mpi"
echo "======================================================================"
for N in $SIZES; do
    BIN="images_N${N}.bin"
    for T in $THREADS; do
        OUT="out_omp${T}_N${N}.txt"
        echo "Running OpenMP (threads=$T, schedule=$SCHEDULE) for N=$N..."
        check "Run omp T=$T N=$N" ./omp "$BIN" "$TEMPLATE_FILE" "$N" "$OUT" "$T" "$SCHEDULE"
    done

    if [ "$HAS_MPI" -eq 1 ]; then
        for P in $MPI_PROCS; do
            OUT="out_mpi${P}_N${N}.txt"
            echo "Running MPI (processes=$P) for N=$N..."
            check "Run mpi P=$P N=$N" mpirun --oversubscribe -np "$P" ./mpi "$BIN" "$TEMPLATE_FILE" "$N" "$OUT"
        done
    else
        echo "MPI execution skipped for N=$N (mpicxx/mpirun not available)."
    fi
done

# ==============================================================================
# Stage 6: Compare each output with the sequential one
# ==============================================================================
echo ""
echo "======================================================================"
echo "Stage 6: Compare each output with the sequential one"
echo "======================================================================"
for N in $SIZES; do
    SEQ_OUT="out_seq_N${N}.txt"
    for T in $THREADS; do
        OMP_OUT="out_omp${T}_N${N}.txt"
        echo "Comparing OpenMP (T=$T, N=$N) output with sequential output..."
        check "Compare omp T=$T N=$N vs seq" $PYTHON verify.py compare "$SEQ_OUT" "$OMP_OUT"
    done

    if [ "$HAS_MPI" -eq 1 ]; then
        for P in $MPI_PROCS; do
            MPI_OUT="out_mpi${P}_N${N}.txt"
            echo "Comparing MPI (P=$P, N=$N) output with sequential output..."
            check "Compare mpi P=$P N=$N vs seq" $PYTHON verify.py compare "$SEQ_OUT" "$MPI_OUT"
        done
    fi
done

# ==============================================================================
# Stage 7: Run benchmark
# ==============================================================================
echo ""
echo "======================================================================"
echo "Stage 7: Run benchmark"
echo "======================================================================"
if [ -f "$BENCHMARK_SCRIPT" ]; then
    echo "Executing benchmark script ($BENCHMARK_SCRIPT)..."
    chmod +x "$BENCHMARK_SCRIPT"
    check "Execute benchmark suite ($BENCHMARK_SCRIPT)" bash "$BENCHMARK_SCRIPT"
else
    echo "Warning: $BENCHMARK_SCRIPT not found. Skipping benchmark execution."
fi

# ==============================================================================
# Stage 8: Generate graphs
# ==============================================================================
echo ""
echo "======================================================================"
echo "Stage 8: Generate graphs"
echo "======================================================================"
if [ -f "$RESULTS_CSV" ]; then
    echo "Generating performance graphs from $RESULTS_CSV via plot.py..."
    check "Plot generation (plot.py)" $PYTHON plot.py
else
    echo "Warning: $RESULTS_CSV not found. Skipping graph generation."
fi

# ==============================================================================
# Final Summary
# ==============================================================================
echo ""
echo "======================================================================"
echo "                         FINAL SUMMARY"
echo "======================================================================"
echo "  Total checks passed : $pass_count"
echo "  Total checks failed : $fail_count"
echo "======================================================================"
if [ "$fail_count" -eq 0 ]; then
    echo "  OVERALL STATUS: ALL CHECKS PASSED [PASS]"
    echo "======================================================================"
    exit 0
else
    echo "  OVERALL STATUS: PIPELINE FAILED [FAIL]"
    echo "======================================================================"
    exit 1
fi
