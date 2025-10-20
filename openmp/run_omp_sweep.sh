#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_omp_sweep.sh dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [out=results.csv]
DATA="${1:-dados.csv}"
CINIT="${2:-centroides_iniciais.csv}"
MAXIT="${3:-50}"
EPS="${4:-1e-4}"
OUT="${5:-results.csv}"

# Compile
gcc -O2 -fopenmp -std=c99 kmeans_1d_omp.c -o kmeans_1d_omp -lm

# Detect logical CPUs (Linux/macOS)
if command -v nproc >/dev/null 2>&1; then
  NPROC=$(nproc --all)
elif command -v sysctl >/dev/null 2>&1; then
  NPROC=$(sysctl -n hw.logicalcpu)
else
  NPROC=8
fi

# Thread grid (cap at NPROC, use powers of two up to 16 by default)
THREADS="1 2 4 8 16"
TGRID=""
for T in $THREADS; do
  if [ "$T" -le "$NPROC" ]; then TGRID="$TGRID $T"; fi
done

SCHEDULES="static dynamic"
CHUNKS="0 1 8 32"

# Header
echo "T,schedule,chunk,N,K,max_iter,eps,iterations,ms,sse,monotonic,speedup,points_per_sec" > "$OUT"

# ---- Baseline (T = 1) ----
BEST_MS=
BEST_LINE=
BEST_S="-"
BEST_CH="-"

for S in $SCHEDULES; do
  for CH in $CHUNKS; do
    LINE=$(OMP_NUM_THREADS=1 ./kmeans_1d_omp "$DATA" "$CINIT" "$MAXIT" "$EPS" - - "$S" "$CH" --csv)
    # Program CSV: N,K,max_iter,eps,T_used,schedule,chunk,iterations,ms,sse,monotonic
    MS=$(echo "$LINE" | awk -F, '{print $(NF-2)}')
    TUSED=$(echo "$LINE" | awk -F, '{print $5}')

    if [ "$TUSED" != "1" ]; then
      echo "Aviso: baseline esperado com T=1, mas obtido T=$TUSED" >&2
    fi

    # Pick the minimum ms
    if [ -z "${BEST_MS:-}" ] || awk "BEGIN{exit !($MS < $BEST_MS)}"; then
      BEST_MS="$MS"
      BEST_LINE="$LINE"
      BEST_S="$S"
      BEST_CH="$CH"
    fi
  done
done

# Compute throughput for baseline
# Reorder fields to match our header and compute derived columns
# Output fields: T,schedule,chunk,N,K,max_iter,eps,iterations,ms,sse,monotonic,speedup,points_per_sec
{
  IFS=, read -r N K MAXIT_USED EPS_USED TUSED SCHED_USED CHUNK_USED ITERS MS SSE MONO <<< "$BEST_LINE"
  PPS=$(awk -v n="$N" -v ms="$MS" 'BEGIN{ if(ms>0){printf "%.6f", n*1000.0/ms} else {printf "0.000000"} }')
  echo "$TUSED,$SCHED_USED,$CHUNK_USED,$N,$K,$MAXIT_USED,$EPS_USED,$ITERS,$MS,$SSE,$MONO,1.000000,$PPS"
} >> "$OUT"

# ---- Sweep T > 1 ----
for T in $TGRID; do
  if [ "$T" -eq 1 ]; then continue; fi
  for S in $SCHEDULES; do
    for CH in $CHUNKS; do
      LINE=$(OMP_NUM_THREADS=$T ./kmeans_1d_omp "$DATA" "$CINIT" "$MAXIT" "$EPS" - - "$S" "$CH" --csv)
      N=$(echo "$LINE" | awk -F, '{print $1}')
      K=$(echo "$LINE" | awk -F, '{print $2}')
      MAXIT_USED=$(echo "$LINE" | awk -F, '{print $3}')
      EPS_USED=$(echo "$LINE" | awk -F, '{print $4}')
      TUSED=$(echo "$LINE" | awk -F, '{print $5}')
      SCHED_USED=$(echo "$LINE" | awk -F, '{print $6}')
      CHUNK_USED=$(echo "$LINE" | awk -F, '{print $7}')
      ITERS=$(echo "$LINE" | awk -F, '{print $8}')
      MS=$(echo "$LINE" | awk -F, '{print $9}')
      SSE=$(echo "$LINE" | awk -F, '{print $10}')
      MONO=$(echo "$LINE" | awk -F, '{print $11}')

      SPD=$(python3 - <<PY
base=${BEST_MS}
ms=${MS}
print("{:.6f}".format(base/ms if ms>0 else 0.0))
PY
)
      PPS=$(awk -v n="$N" -v ms="$MS" 'BEGIN{ if(ms>0){printf "%.6f", n*1000.0/ms} else {printf "0.000000"} }')
      echo "$TUSED,$SCHED_USED,$CHUNK_USED,$N,$K,$MAXIT_USED,$EPS_USED,$ITERS,$MS,$SSE,$MONO,$SPD,$PPS" >> "$OUT"
    done
  done
done

echo "OK: resultados em $OUT"
