# K-means 1D com Paralelização Progressiva

Este projeto consiste na implementação do algoritmo K-means para dados unidimensionais (1D), com uma versão sequencial e uma versão paralela utilizando a API OpenMP. O objetivo é analisar o impacto do paralelismo no desempenho do algoritmo em diferentes cenários.

Este trabalho foi desenvolvido para a disciplina de Programação Concorrente e Distribuída da Universidade Federal de São Paulo (UNIFESP).

## Estrutura do Projeto

O repositório está organizado da seguinte forma:

```
.
├── gerardados/             # Conjuntos de dados para teste
│   ├── 104/                # Contém 10^4 (10 mil) pontos de dados
│   ├── 105/                # Contém 10^5 (100 mil) pontos de dados
│   └── ...
├── openmp/                 # Implementação paralela com OpenMP
│   ├── kmeans_1d_naive.c   # Código-fonte da versão OpenMP
│   └── run_omp_sweep.sh    # Script para automatizar testes de desempenho
├── serial/                 # (Futura) Implementação sequencial
├── cuda/                   # (Futura) Implementação com CUDA
├── mpi/                    # (Futura) Implementação com MPI
└── gera_dados.py           # Script para gerar novos conjuntos de dados
```

## Versão OpenMP

A implementação paralela com OpenMP foi projetada para avaliar o ganho de desempenho (speedup) em relação a uma execução sequencial.

### Pré-requisitos

Para compilar e executar esta versão, você precisará de:

  - Um compilador C com suporte a OpenMP (ex: **GCC**)
  - `bash` para executar o script de análise

### Compilação

Para compilar o programa, navegue até a pasta `openmp` e execute o comando:

```bash
cd openmp
gcc -O2 -fopenmp -std=c99 kmeans_1d_naive.c -o kmeans_1d_naive -lm
```

**Observação:** A flag `-fopenmp` é essencial para habilitar as diretivas de paralelismo do OpenMP.

### Execução e Análise de Desempenho

Para facilitar a análise de desempenho, foi criado o script `run_omp_sweep.sh`. Este script automatiza a execução do programa com diferentes configurações de paralelismo para encontrar a combinação mais eficiente.

**O que o script faz?**

1.  **Compila** o código-fonte.
2.  **Executa um baseline**: Mede o tempo de execução com apenas 1 thread para ter uma referência.
3.  **Realiza um "sweep" (varredura)**: Testa o programa com múltiplas combinações de:
      * Número de threads (ex: 2, 4, 8, ...).
      * Estratégias de agendamento do OpenMP (`static`, `dynamic`).
      * Tamanhos de `chunk` (divisão de tarefas).
4.  **Gera um relatório**: Salva todos os resultados em um arquivo `.csv`, contendo o tempo de execução, o **speedup** (aceleração em relação ao baseline) e a vazão (pontos por segundo) para cada teste.

**Como usar o script:**

O script deve ser executado de dentro da pasta `openmp`. Ele recebe como argumento os caminhos para os arquivos de dados e de centroides.

```bash
# Exemplo para testar com o conjunto de dados de 100 mil pontos (10^5)
./run_omp_sweep.sh ../gerardados/105/dados.csv ../gerardados/105/centroides_iniciais.csv
```

Após a execução, um arquivo chamado `results.csv` será criado na pasta `openmp` com todos os dados de desempenho para análise.
