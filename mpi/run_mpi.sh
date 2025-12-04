#!/bin/bash
# Script para compilar e rodar o K-means MPI

# Compilar
echo "Compilando kmeans_1d_mpi.c..."
mpicc -O2 kmeans_1d_mpi.c -o kmeans_1d_mpi -lm

if [ $? -ne 0 ]; then
    echo "Erro na compilação."
    exit 1
fi

echo "Compilação OK."

# Arquivo de resultados
OUT="results_mpi_final.csv"
echo "N,K,max_iter,eps,P,iterations,ms,sse,throughput" > "$OUT"

# Datasets para testar
SIZES=("104" "105" "106")
PROCS="1 2 4"

for SZ in "${SIZES[@]}"; do
    DATA_DIR="../gerardados/$SZ"
    DADOS="$DATA_DIR/dados.csv"
    CENTROIDES="$DATA_DIR/centroides_iniciais.csv"

    if [ ! -f "$DADOS" ]; then
        echo "Aviso: Dataset $SZ não encontrado em $DADOS. Pulando..."
        continue
    fi

    echo "=== Processando Dataset 10^$SZ ==="
    
    for P in $PROCS; do
        echo "  Running with P=$P..."
        # Executa e anexa a saída CSV ao arquivo de resultados
        # O programa C já imprime a linha CSV quando passamos --csv
        mpirun -np $P ./kmeans_1d_mpi "$DADOS" "$CENTROIDES" 50 1e-4 "assign_mpi_${SZ}_${P}.csv" "centroids_mpi_${SZ}_${P}.csv" --csv >> "$OUT"
    done
done

echo "Concluído. Resultados em $OUT"
