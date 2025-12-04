import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Nome do arquivo de entrada
INPUT_FILE = 'results_cuda_final.csv'

if not os.path.exists(INPUT_FILE):
    print(f"Erro: Arquivo {INPUT_FILE} não encontrado.")
    exit(1)

# Carregar os dados
df = pd.read_csv(INPUT_FILE)

# Identificar os tamanhos de dataset (N) únicos
datasets = df['N'].unique()
datasets.sort()

print(f"Datasets encontrados (N): {datasets}")

for N in datasets:
    print(f"Gerando gráficos para N={N}...")
    
    # Filtrar dados para este dataset
    df_subset = df[df['N'] == N].copy()
    
    # Ordenar por BlockSize para consistência no eixo X
    df_subset.sort_values(by='blockSize', inplace=True)
    
    # Criar figura com subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'CUDA K-means Performance Analysis (N={N})', fontsize=16, fontweight='bold')

    # Configurar labels para o eixo x
    x_labels = [f'BS={bs}\nGS={gs}' for bs, gs in zip(df_subset['blockSize'], df_subset['gridSize'])]
    x_pos = np.arange(len(df_subset))

    # 1. Tempo Total vs Configuração
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x_pos, df_subset['ms_total'], color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Configuração (Block Size / Grid Size)', fontweight='bold')
    ax1.set_ylabel('Tempo Total (ms)', fontweight='bold')
    ax1.set_title('Tempo Total de Execução', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    # Adicionar valores nas barras
    for i, (bar, val) in enumerate(zip(bars1, df_subset['ms_total'])):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (val*0.01), 
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # 2. Decomposição dos Tempos
    ax2 = axes[0, 1]
    width = 0.25
    x_offset = np.arange(len(df_subset))
    bars2_1 = ax2.bar(x_offset - width, df_subset['ms_kernel'], width, label='Kernel', color='coral', alpha=0.8, edgecolor='black')
    bars2_2 = ax2.bar(x_offset, df_subset['ms_h2d'], width, label='Host→Device', color='lightgreen', alpha=0.8, edgecolor='black')
    bars2_3 = ax2.bar(x_offset + width, df_subset['ms_d2h'], width, label='Device→Host', color='plum', alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Configuração (Block Size / Grid Size)', fontweight='bold')
    ax2.set_ylabel('Tempo (ms)', fontweight='bold')
    ax2.set_title('Decomposição dos Tempos', fontweight='bold')
    ax2.set_xticks(x_offset)
    ax2.set_xticklabels(x_labels, fontsize=9)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # 3. Throughput (Pontos por Segundo)
    ax3 = axes[1, 0]
    bars3 = ax3.bar(x_pos, df_subset['points_per_sec'], color='mediumseagreen', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Configuração (Block Size / Grid Size)', fontweight='bold')
    ax3.set_ylabel('Pontos/Segundo', fontweight='bold')
    ax3.set_title('Throughput de Processamento', fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(x_labels, fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    # Adicionar valores nas barras
    for i, (bar, val) in enumerate(zip(bars3, df_subset['points_per_sec'])):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (val*0.01), 
                 f'{val/1e6:.2f}M', ha='center', va='bottom', fontsize=9)

    # 4. Comparação de Eficiência (Percentual)
    ax4 = axes[1, 1]
    # Calcular porcentagem de cada componente
    total_time = df_subset['ms_total']
    # Evitar divisão por zero
    total_time = total_time.replace(0, 1)
    
    kernel_pct = (df_subset['ms_kernel'] / total_time) * 100
    h2d_pct = (df_subset['ms_h2d'] / total_time) * 100
    d2h_pct = (df_subset['ms_d2h'] / total_time) * 100

    x_stacked = np.arange(len(df_subset))
    ax4.bar(x_stacked, kernel_pct, label='Kernel', color='coral', alpha=0.8, edgecolor='black')
    ax4.bar(x_stacked, h2d_pct, bottom=kernel_pct, label='Host→Device', color='lightgreen', alpha=0.8, edgecolor='black')
    ax4.bar(x_stacked, d2h_pct, bottom=kernel_pct+h2d_pct, label='Device→Host', color='plum', alpha=0.8, edgecolor='black')
    ax4.set_xlabel('Configuração (Block Size / Grid Size)', fontweight='bold')
    ax4.set_ylabel('Porcentagem (%)', fontweight='bold')
    ax4.set_title('Distribuição Percentual dos Tempos', fontweight='bold')
    ax4.set_xticks(x_stacked)
    ax4.set_xticklabels(x_labels, fontsize=9)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    
    # Nome do arquivo baseado no tamanho N (ex: 10000 -> 10^4)
    exponent = int(np.log10(N))
    filename = f'cuda_analysis_10{exponent}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Gráfico salvo como '{filename}'")
    plt.close()

    # Imprimir estatísticas resumidas para este N
    print(f"  [Resumo N={N}]")
    best_idx = df_subset['ms_total'].idxmin()
    print(f"    Melhor Config: BS={df_subset.loc[best_idx, 'blockSize']} | Tempo={df_subset.loc[best_idx, 'ms_total']:.3f} ms")
    print("-" * 40)

print("\nProcessamento concluído.")
