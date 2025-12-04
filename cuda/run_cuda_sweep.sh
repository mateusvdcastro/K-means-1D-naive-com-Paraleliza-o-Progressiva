#!/bin/bash
set -euo pipefail

# --- Configuração ---
# Ajuste a arquitetura conforme sua GPU (sm_75 = T4/RTX20xx, sm_86 = RTX30xx, sm_89 = RTX40xx)
ARCH="sm_89"
OUT="results_cuda_final.csv"

# 1. Compilar
echo "--- Compilando kmeans_1d_cuda.cu ($ARCH) ---"
nvcc -arch=$ARCH -O2 -o kmeans_cuda kmeans_1d_cuda.cu

# 2. Preparar CSV de Saída
echo "blockSize,gridSize,N,K,max_iter,eps,iterations,ms_total,ms_kernel,ms_h2d,ms_d2h,sse,monotonic,points_per_sec" > "$OUT"

# 3. Definição dos Testes
SIZES=("104" "105" "106")
BLOCK_SIZES="128 256 512"

# 4. Loop de Execução
for SZ in "${SIZES[@]}"; do
    DATA_DIR="../gerardados/$SZ"
    DADOS="$DATA_DIR/dados.csv"
    CENTROIDES="$DATA_DIR/centroides_iniciais.csv"

    if [ ! -f "$DADOS" ]; then
        echo "ERRO: Arquivo $DADOS não encontrado!"
        continue
    fi

    echo "=== Dataset 10^$SZ ==="

    for BS in $BLOCK_SIZES; do
        echo "  -> BlockSize: $BS"
        
        # Executa o binário CUDA
        # Argumentos: dados, centroides, iter, eps, out_assign, out_centroids, blockSize
        OUTPUT=$(./kmeans_cuda "$DADOS" "$CENTROIDES" 50 1e-4 "-" "-" "$BS")

        # --- Parsing dos Resultados ---
        # Extrai os valores da saída padrão do programa C
        # Exemplo de linha esperada: "Iterações: 15 | SSE final: ... | Tempo Total: ..."
        
        # Pega N e K
        NK_LINE=$(echo "$OUTPUT" | grep "^N=")
        N=$(echo "$NK_LINE" | cut -d' ' -f1 | cut -d'=' -f2)
        K=$(echo "$NK_LINE" | cut -d' ' -f2 | cut -d'=' -f2)
        
        # Pega GridSize
        GRID=$(echo "$OUTPUT" | grep "^GRID:" | cut -d' ' -f2)

        # Pega Métricas
        RES_LINE=$(echo "$OUTPUT" | grep "^Iterações:")
        ITERS=$(echo "$RES_LINE" | cut -d'|' -f1 | cut -d':' -f2 | tr -d ' ')
        SSE=$(echo "$RES_LINE"   | cut -d'|' -f2 | cut -d':' -f2 | tr -d ' ')
        MS_TOT=$(echo "$RES_LINE"| cut -d'|' -f3 | cut -d':' -f2 | cut -d' ' -f2)
        MS_KER=$(echo "$RES_LINE"| cut -d'|' -f4 | cut -d':' -f2 | cut -d' ' -f2)
        MS_H2D=$(echo "$RES_LINE"| cut -d'|' -f5 | cut -d':' -f2 | cut -d' ' -f2)
        MS_D2H=$(echo "$RES_LINE"| cut -d'|' -f6 | cut -d':' -f2 | cut -d' ' -f2)
        MONO=$(echo "$RES_LINE"  | cut -d'|' -f7 | cut -d':' -f2 | tr -d ' ')

        # Calcula Throughput
        PPS=$(awk -v n="$N" -v ms="$MS_TOT" 'BEGIN{ if(ms>0) printf "%.2f", (n*1000.0)/ms; else print 0 }')

        # Salva no CSV
        echo "$BS,$GRID,$N,$K,50,1e-4,$ITERS,$MS_TOT,$MS_KER,$MS_H2D,$MS_D2H,$SSE,$MONO,$PPS" >> "$OUT"
    done
done

echo "--- Concluído. Resultados em $OUT ---"