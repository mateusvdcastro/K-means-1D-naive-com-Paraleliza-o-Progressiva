#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_omp_sweep.sh dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4]
DATA="${1:-dados.csv}"
CINIT="${2:-centroides_iniciais.csv}"
MAXIT="${3:-50}"
EPS="${4:-1e-4}"

# Compile
gcc -O2 -fopenmp -std=c99 kmeans_1d_omp.c -o kmeans_1d_omp -lm

# Detect logical CPUs (Linux/macOS)
if command -v nproc >/dev/null 2>&1; then
  NPROC=$(nproc --all)
else
  NPROC=$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 8)
fi

THREADS=(1)
t=1
while [ $t -lt "$NPROC" ]; do
  t=$(( t * 2 ))
  THREADS+=($t)
  if [ $t -ge $(( NPROC * 2 )) ]; then break; fi
done

SCHEDULES=("static" "dynamic")
CHUNKS=(0 1 8 32)

# Warm-up
export OMP_NUM_THREADS=1
./kmeans_1d_omp "$DATA" "$CINIT" "$MAXIT" "$EPS" - - static 0 >/dev/null

# Header
OUT="results.csv"
echo "T,schedule,chunk,N,K,max_iter,eps,iterations,ms,sse,monotonic,speedup" > "$OUT"

# Baseline with T=1
export OMP_NUM_THREADS=1
BASE=$(./kmeans_1d_omp "$DATA" "$CINIT" "$MAXIT" "$EPS" - - static 0 --csv)
BASE_MS=$(echo "$BASE" | awk -F, '{print $(NF-2)}')
echo "1,static,0,$(echo "$BASE" | cut -d, -f1-4),$(echo "$BASE" | cut -d, -f8-10),$(echo "$BASE" | awk -F, '{print $NF}'),1.0" >> "$OUT"

# Sweep
for T in "${THREADS[@]}"; do
  for S in "${SCHEDULES[@]}"; do
    for CH in "${CHUNKS[@]}"; do
      export OMP_NUM_THREADS=$T
      LINE=$(./kmeans_1d_omp "$DATA" "$CINIT" "$MAXIT" "$EPS" - - "$S" "$CH" --csv)
      MS=$(echo "$LINE" | awk -F, '{print $(NF-2)}')
      SPD=$(python3 - <<PY
base=${BASE_MS}
ms=${MS}
print("{:.6f}".format(base/ms if ms>0 else 0.0))
PY
)
      echo "$T,$S,$CH,$(echo "$LINE" | cut -d, -f1-4),$(echo "$LINE" | cut -d, -f8-10),$(echo "$LINE" | awk -F, '{print $NF}'),$SPD" >> "$OUT"
    done
  done
done

echo "OK: resultados em $OUT"
