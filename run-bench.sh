#!/bin/bash
set -e

N_ITER=${1:-5000}

echo "=== Building benchmark ==="
gcc -O3 -march=native -I include -L lib bench-even-odd.c -lceed -lm -o bench-even-odd -Wl,-rpath,lib

echo ""
echo "=== Baseline: /cpu/self/opt/blocked (no even-odd) ==="
for ncomp in 1 3; do
  echo ""
  echo "--- ncomp=$ncomp ---"
  ./bench-even-odd /cpu/self/opt/blocked $ncomp $N_ITER
done

echo ""
echo "=== Even-Odd: /cpu/self/avx/blocked ==="
for ncomp in 1 3; do
  echo ""
  echo "--- ncomp=$ncomp ---"
  ./bench-even-odd /cpu/self/avx/blocked $ncomp $N_ITER
done
