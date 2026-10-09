#!/usr/bin/env bash
# benchmark.sh -- Step 6: Run all versions, 5 reps each, log to results.csv
#
# Usage: chmod +x benchmark.sh && ./benchmark.sh
#
# Appends rows to results.csv:
#   version, N, threads_or_procs, run, load_time, compute_time
#
# Parses "Load time : X s" and "Compute time : X s" from program stdout.

set -euo pipefail

CSV=results.csv
REPS=5

# Write CSV header if file does not exist
if [ ! -f "$CSV" ]; then
    echo "version,N,threads_or_procs,run,load_time,compute_time" > "$CSV"
fi

# Helper: run a command, capture stdout, parse the two timing lines
run_and_log() {
    local version="$1"
    local N="$2"
    local tp="$3"
    local run_id="$4"
    shift 4
    local out
    out=$("$@" 2>&1)
    local load_t compute_t
    load_t=$(echo "$out"    | grep -oP '(?<=Load time    : )[\d.]+' || echo "NA")
    compute_t=$(echo "$out" | grep -oP '(?<=Compute time : )[\d.]+' || echo "NA")
    echo "$version,$N,$tp,$run_id,$load_t,$compute_t" >> "$CSV"
    echo "  [$version N=$N tp=$tp run=$run_id]  load=${load_t}s  compute=${compute_t}s"
}

for N in 1000 5000 10000 50000; do
    BIN="images_N${N}.bin"
    echo ""
    echo "====== N = ${N} ======"

    # -- Sequential --
    for run in $(seq 1 $REPS); do
        run_and_log "seq" "$N" 1 "$run" \
            ./seq "$BIN" template.bin "$N" /dev/null
    done

    # -- OpenMP (1, 2, 4, 8 threads) --
    for T in 1 2 4 8; do
        for run in $(seq 1 $REPS); do
            OMP_SCHEDULE=dynamic run_and_log "omp" "$N" "$T" "$run" \
                ./omp "$BIN" template.bin "$N" /dev/null "$T" dynamic
        done
    done

    # -- MPI (1, 2, 4, 8 processes) --
    for P in 1 2 4 8; do
        for run in $(seq 1 $REPS); do
            run_and_log "mpi" "$N" "$P" "$run" \
                mpirun -np "$P" ./mpi "$BIN" template.bin "$N" /dev/null
        done
    done
done

echo ""
echo "Benchmark complete. Results in $CSV"
