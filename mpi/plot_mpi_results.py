import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import io

# Nome do arquivo de entrada
INPUT_FILE = 'results_mpi_final.csv'

if not os.path.exists(INPUT_FILE):
    print(f"Erro: Arquivo {INPUT_FILE} não encontrado.")
    exit(1)

# Ler o arquivo e filtrar linhas inválidas (que começam com "Processo")
valid_lines = []
with open(INPUT_FILE, 'r') as f:
    header = f.readline()
    valid_lines.append(header)
    for line in f:
        if not line.strip().startswith("Processo"):
            valid_lines.append(line)

# Criar DataFrame a partir das linhas válidas
csv_content = "".join(valid_lines)
df = pd.read_csv(io.StringIO(csv_content))

# Converter colunas numéricas (caso haja algum problema de tipo)
numeric_cols = ['N', 'K', 'max_iter', 'eps', 'P', 'iterations', 'ms', 'sse', 'throughput']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Identificar os tamanhos de dataset (N) únicos
datasets = df['N'].unique()
datasets.sort()

print(f"Datasets encontrados (N): {datasets}")

for N in datasets:
    print(f"Gerando gráficos para N={N}...")
    
    # Filtrar dados para este dataset
    df_subset = df[df['N'] == N].copy()
    
    # Ordenar por P (número de processos)
    df_subset.sort_values(by='P', inplace=True)
    
    # Calcular Speedup
    # Speedup(P) = Tempo(1) / Tempo(P)
    # Assumindo que existe P=1. Se não existir, usar o menor P como base ou avisar.
    baseline_row = df_subset[df_subset['P'] == 1]
    if not baseline_row.empty:
        baseline_time = baseline_row.iloc[0]['ms']
        df_subset['speedup'] = baseline_time / df_subset['ms']
    else:
        print(f"Aviso: Não encontrado resultado para P=1 com N={N}. Speedup não será calculado corretamente.")
        df_subset['speedup'] = np.nan

    # Criar figura com subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'MPI K-means Performance Analysis (N={N})', fontsize=16, fontweight='bold')

    # 1. Tempo de Execução vs P
    ax1 = axes[0]
    ax1.plot(df_subset['P'], df_subset['ms'], marker='o', linestyle='-', color='b', label='Tempo (ms)')
    ax1.set_xlabel('Número de Processos (P)', fontweight='bold')
    ax1.set_ylabel('Tempo (ms)', fontweight='bold')
    ax1.set_title('Tempo de Execução', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(df_subset['P'])

    # 2. Speedup vs P
    ax2 = axes[1]
    if not df_subset['speedup'].isna().all():
        ax2.plot(df_subset['P'], df_subset['speedup'], marker='s', linestyle='-', color='g', label='Speedup')
        # Linha ideal
        ax2.plot(df_subset['P'], df_subset['P'], linestyle='--', color='gray', label='Ideal')
        ax2.set_xlabel('Número de Processos (P)', fontweight='bold')
        ax2.set_ylabel('Speedup', fontweight='bold')
        ax2.set_title('Speedup', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(df_subset['P'])

    # 3. Throughput vs P
    ax3 = axes[2]
    ax3.plot(df_subset['P'], df_subset['throughput'], marker='^', linestyle='-', color='r', label='Throughput')
    ax3.set_xlabel('Número de Processos (P)', fontweight='bold')
    ax3.set_ylabel('Throughput (pontos/s)', fontweight='bold')
    ax3.set_title('Throughput', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(df_subset['P'])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_filename = f'mpi_performance_N_{N}.png'
    plt.savefig(output_filename)
    print(f"Gráfico salvo em: {output_filename}")
    plt.close()

print("Concluído.")
