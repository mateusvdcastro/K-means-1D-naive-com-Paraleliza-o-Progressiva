# K-means 1D - Implementação MPI

Esta pasta contém a implementação do algoritmo K-means 1D utilizando MPI (Message Passing Interface) para memória distribuída.

## Estrutura

- `kmeans_1d_mpi.c`: Código fonte em C com MPI.
- `run_mpi.sh`: Script para compilar e executar testes.

## Pré-requisitos

- Compilador MPI (ex: `mpicc` do OpenMPI ou MPICH).
- `mpirun` ou `mpiexec` para execução.

## Compilação

Para compilar manualmente:

```bash
mpicc -O2 kmeans_1d_mpi.c -o kmeans_1d_mpi -lm
```

## Execução

Para rodar com P processos:

```bash
mpirun -np <P> ./kmeans_1d_mpi <dados.csv> <centroides.csv> [max_iter] [eps] [assign.csv] [centroids.csv] [--csv]
```

Exemplo:

```bash
mpirun -np 4 ./kmeans_1d_mpi ../dados.csv ../centroides_iniciais.csv 50 1e-4 assign.csv centroids.csv
```

## Script de Automação

O script `run_mpi.sh` compila o código e executa testes com 1, 2 e 4 processos.

```bash
./run_mpi.sh
```