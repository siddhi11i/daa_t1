# Makefile for Parallel Template Matching project
# Run on Linux/WSL: make all
#
# Targets
# -------
#   all          - build seq, omp, mpi  (+ existing match_* versions)
#   seq          - sequential version (seq.cpp)
#   omp          - OpenMP version (omp.cpp)
#   mpi          - MPI version (mpi.cpp)
#   legacy       - build original match_serial, match_omp, match_pthread, match_mpi
#   clean        - remove all binaries

CXX      = g++
MPICXX   = mpicxx
CXXFLAGS = -O3 -Wall

.PHONY: all seq omp mpi legacy clean

all: seq omp mpi

# ---- Step 2: sequential (spec-named) ----------------------------------------
seq: seq.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# ---- Step 4: OpenMP (spec-named) --------------------------------------------
omp: omp.cpp
	$(CXX) $(CXXFLAGS) -fopenmp -o $@ $<

# ---- Step 5: MPI (spec-named) -----------------------------------------------
mpi: mpi.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $<

# ---- Legacy versions (match_*.cpp) ------------------------------------------
legacy: match_serial match_omp match_pthread match_mpi

match_serial: match_serial.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

match_omp: match_omp.cpp
	$(CXX) $(CXXFLAGS) -fopenmp -o $@ $<

match_pthread: match_pthread.cpp
	$(CXX) $(CXXFLAGS) -pthread -o $@ $<

match_mpi: match_mpi.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f seq omp mpi match_serial match_omp match_pthread match_mpi
