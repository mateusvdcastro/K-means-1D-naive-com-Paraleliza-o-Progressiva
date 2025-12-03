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

# Definir caminhos dos dados (ajuste conforme necessário)
# Exemplo usando os dados gerados na pasta ../gerardados/104
DATA_DIR="../gerardados/104"
DADOS="$DATA_DIR/dados.csv"
CENTROIDES="$DATA_DIR/centroides_iniciais.csv"

if [ ! -f "$DADOS" ]; then
    echo "Arquivo de dados não encontrado: $DADOS"
    echo "Tentando usar dados da raiz..."
    DADOS="../dados.csv"
    CENTROIDES="../centroides_iniciais.csv"
fi

if [ ! -f "$DADOS" ]; then
    echo "Arquivo de dados não encontrado. Gere os dados primeiro."
    exit 1
fi

# Rodar com diferentes números de processos
echo "Rodando K-means MPI..."
echo "N,K,max_iter,eps,P,iterations,ms,sse"

for P in 1 2 4; do
    mpirun -np $P ./kmeans_1d_mpi "$DADOS" "$CENTROIDES" 50 1e-4 "assign_mpi_${P}.csv" "centroids_mpi_${P}.csv" --csv
done
