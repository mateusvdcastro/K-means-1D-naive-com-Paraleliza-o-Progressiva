import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Carregar os dados
df = pd.read_csv('cuda_sweep_results.csv')

# Criar figura com subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('CUDA K-means Performance Analysis', fontsize=16, fontweight='bold')

# Configurar labels para o eixo x
x_labels = [f'BS={bs}\nGS={gs}' for bs, gs in zip(df['blockSize'], df['gridSize'])]
x_pos = np.arange(len(df))

# 1. Tempo Total vs Configuração
ax1 = axes[0, 0]
bars1 = ax1.bar(x_pos, df['ms_total'], color='steelblue', alpha=0.7, edgecolor='black')
ax1.set_xlabel('Configuração (Block Size / Grid Size)', fontweight='bold')
ax1.set_ylabel('Tempo Total (ms)', fontweight='bold')
ax1.set_title('Tempo Total de Execução', fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(x_labels, fontsize=9)
ax1.grid(axis='y', alpha=0.3)
# Adicionar valores nas barras
for i, (bar, val) in enumerate(zip(bars1, df['ms_total'])):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
             f'{val:.3f}', ha='center', va='bottom', fontsize=9)

# 2. Decomposição dos Tempos
ax2 = axes[0, 1]
width = 0.25
x_offset = np.arange(len(df))
bars2_1 = ax2.bar(x_offset - width, df['ms_kernel'], width, label='Kernel', color='coral', alpha=0.8, edgecolor='black')
bars2_2 = ax2.bar(x_offset, df['ms_h2d'], width, label='Host→Device', color='lightgreen', alpha=0.8, edgecolor='black')
bars2_3 = ax2.bar(x_offset + width, df['ms_d2h'], width, label='Device→Host', color='plum', alpha=0.8, edgecolor='black')
ax2.set_xlabel('Configuração (Block Size / Grid Size)', fontweight='bold')
ax2.set_ylabel('Tempo (ms)', fontweight='bold')
ax2.set_title('Decomposição dos Tempos', fontweight='bold')
ax2.set_xticks(x_offset)
ax2.set_xticklabels(x_labels, fontsize=9)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 3. Throughput (Pontos por Segundo)
ax3 = axes[1, 0]
bars3 = ax3.bar(x_pos, df['points_per_sec'], color='mediumseagreen', alpha=0.7, edgecolor='black')
ax3.set_xlabel('Configuração (Block Size / Grid Size)', fontweight='bold')
ax3.set_ylabel('Pontos/Segundo', fontweight='bold')
ax3.set_title('Throughput de Processamento', fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(x_labels, fontsize=9)
ax3.grid(axis='y', alpha=0.3)
# Adicionar valores nas barras
for i, (bar, val) in enumerate(zip(bars3, df['points_per_sec'])):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50000, 
             f'{val/1e6:.2f}M', ha='center', va='bottom', fontsize=9)

# 4. Comparação de Eficiência
ax4 = axes[1, 1]
# Calcular porcentagem de cada componente
total_time = df['ms_total']
kernel_pct = (df['ms_kernel'] / total_time) * 100
h2d_pct = (df['ms_h2d'] / total_time) * 100
d2h_pct = (df['ms_d2h'] / total_time) * 100

x_stacked = np.arange(len(df))
ax4.bar(x_stacked, kernel_pct, label='Kernel', color='coral', alpha=0.8, edgecolor='black')
ax4.bar(x_stacked, h2d_pct, bottom=kernel_pct, label='Host→Device', color='lightgreen', alpha=0.8, edgecolor='black')
ax4.bar(x_stacked, d2h_pct, bottom=kernel_pct+h2d_pct, label='Device→Host', color='plum', alpha=0.8, edgecolor='black')
ax4.set_xlabel('Configuração (Block Size / Grid Size)', fontweight='bold')
ax4.set_ylabel('Porcentagem (%)', fontweight='bold')
ax4.set_title('Distribuição Percentual dos Tempos', fontweight='bold')
ax4.set_xticks(x_stacked)
ax4.set_xticklabels(x_labels, fontsize=9)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('cuda_sweep_analysis.png', dpi=300, bbox_inches='tight')
print("Gráfico salvo como 'cuda_sweep_analysis.png'")

# Imprimir estatísticas
print("\n" + "="*60)
print("RESUMO DOS RESULTADOS")
print("="*60)
print(f"\nMelhor configuração (menor tempo total):")
best_idx = df['ms_total'].idxmin()
print(f"  Block Size: {df.loc[best_idx, 'blockSize']}")
print(f"  Grid Size: {df.loc[best_idx, 'gridSize']}")
print(f"  Tempo Total: {df.loc[best_idx, 'ms_total']:.3f} ms")
print(f"  Throughput: {df.loc[best_idx, 'points_per_sec']/1e6:.2f} M pontos/seg")

print(f"\nMaior throughput:")
best_throughput_idx = df['points_per_sec'].idxmax()
print(f"  Block Size: {df.loc[best_throughput_idx, 'blockSize']}")
print(f"  Grid Size: {df.loc[best_throughput_idx, 'gridSize']}")
print(f"  Throughput: {df.loc[best_throughput_idx, 'points_per_sec']/1e6:.2f} M pontos/seg")
print(f"  Tempo Total: {df.loc[best_throughput_idx, 'ms_total']:.3f} ms")
print("="*60)

plt.show()
