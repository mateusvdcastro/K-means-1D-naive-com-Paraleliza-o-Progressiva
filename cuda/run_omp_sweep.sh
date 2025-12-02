#!/usr/bin/env bash
set -euo pipefail

# 1. Parâmetros de entrada (default)
DATA="${1:-dados.csv}"
CINIT="${2:-centroides_iniciais.csv}"
MAXIT="${3:-50}"
EPS="${4:-1e-4f}"
OUT="${5:-results_cuda.csv}"

# 2. Compilar (usando o -arch=sm_75)
echo "--- Compilando kmeans_cuda (com -arch=sm_75) ---"
nvcc -arch=sm_89 -o kmeans_cuda kmeans_1d_cuda.cu
if [ $? -ne 0 ]; then
  echo "Falha na compilação!"
  exit 1
fi

# 3. Parâmetros da varredura (Sweep)
# (Você pode adicionar mais tamanhos aqui, ex: "32 64 128 256 512 1024")
BLOCK_SIZES="128 256 512"

# 4. Header do CSV (cabeçalho)
# --- MODIFICAÇÃO: Adicionadas novas colunas ---
mkdir -p "$(dirname "$OUT")"
echo "blockSize,gridSize,N,K,max_iter,eps,iterations,ms_total,ms_kernel,ms_h2d,ms_d2h,sse,monotonic,points_per_sec" > "$OUT"

echo "--- Iniciando varredura (sweep) de Block Size ---"

# 5. Loop de varredura
for BS in $BLOCK_SIZES; do
  echo "Rodando com blockSize = $BS..."

  # Roda o programa e captura a saída
  # O $BS no final é o [blockSize] (argv[7]) que o main() agora lê
  LOG_OUTPUT=$(./kmeans_cuda "$DATA" "$CINIT" "$MAXIT" "$EPS" - - "$BS")

  # 6. ###############################################################
  #  ##     INÍCIO DA SEÇÃO DE PARSING (Modificada)    ##
  #  ###############################################################

  # Captura a linha: N=10000 K=4 max_iter=50 eps=1e-4f
  NK_LINE=$(echo "$LOG_OUTPUT" | grep "^N=")
  N=$(echo "$NK_LINE" | cut -d' ' -f1 | cut -d'=' -f2)
  K=$(echo "$NK_LINE" | cut -d' ' -f2 | cut -d'=' -f2)

  # *** MODIFICAÇÃO: Captura o GridSize ***
  GRID_LINE=$(echo "$LOG_OUTPUT" | grep "^GRID:")
  GRID_SIZE=$(echo "$GRID_LINE" | cut -d' ' -f2)

  # Captura a linha de resultado principal
  # Ex: Iterações: 3 | SSE final: 9930.14 | Tempo Total: 8.460 ms | T. Kernel: 0.123 ms | T. H2D: 0.456 ms | T. D2H: 0.789 ms | Monotônico: sim
  RESULT_LINE=$(echo "$LOG_OUTPUT" | grep "^Iterações:")

  # Extrai cada parte usando 'cut' (cortar) pelo delimitador '|'
  # tr -d ' ' remove espaços em branco

  # *** MODIFICAÇÃO: Lógica de parsing atualizada para os novos campos ***
  ITERS=$(echo "$RESULT_LINE" | cut -d'|' -f1 | cut -d':' -f2 | tr -d ' ')
  SSE=$(echo "$RESULT_LINE"  | cut -d'|' -f2 | cut -d':' -f2 | tr -d ' ')

  # Pega o valor numérico (ex: "8.460") depois de "Tempo Total: "
  MS_TOTAL=$(echo "$RESULT_LINE" | cut -d'|' -f3 | cut -d':' -f2 | cut -d' ' -f2)
  MS_KERNEL=$(echo "$RESULT_LINE" | cut -d'|' -f4 | cut -d':' -f2 | cut -d' ' -f2)
  MS_H2D=$(echo "$RESULT_LINE"  | cut -d'|' -f5 | cut -d':' -f2 | cut -d' ' -f2)
  MS_D2H=$(echo "$RESULT_LINE"  | cut -d'|' -f6 | cut -d':' -f2 | cut -d' ' -f2)
  MONO=$(echo "$RESULT_LINE"   | cut -d'|' -f7 | cut -d':' -f2 | tr -d ' ')

  # 7. Calcular métricas (Pontos/s)
  # (Calculado com base no tempo total)
  PPS=$(awk -v n="$N" -v ms="$MS_TOTAL" 'BEGIN{ if(ms>0){printf "%.6f", n*1000.0/ms} else {printf "0.000000"} }')

  # 8. Salvar no CSV
  # *** MODIFICAÇÃO: Adiciona novas colunas ao CSV ***
  echo "$BS,$GRID_SIZE,$N,$K,$MAXIT,$EPS,$ITERS,$MS_TOTAL,$MS_KERNEL,$MS_H2D,$MS_D2H,$SSE,$MONO,$PPS" >> "$OUT"

  # ###############################################################
  #  ##      FIM DA SEÇÃO DE PARSING           ##
  #  ###############################################################
done

echo "--- Varredura CUDA finalizada. Resultados em $OUT ---"