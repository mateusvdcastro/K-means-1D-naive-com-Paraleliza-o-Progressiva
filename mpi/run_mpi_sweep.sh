set -e

# --- CONFIGURAÇÃO: Caminho do MS-MPI (Windows) ---
export PATH=$PATH:"/c/Program Files/Microsoft MPI/Bin/"

# --- 1. COMPILAÇÃO ---
echo "--- 1. Compilando o codigo MPI ---"
gcc -O2 -std=c99 mpi.c -o kmeans_mpi.exe -lmsmpi -lm

if [ $? -ne 0 ]; then
  echo "ERRO: Falha na compilacao."
  exit 1
fi

# Arquivo de saida
OUT="results_mpi_completo.csv"
mkdir -p "$(dirname "$OUT")"
# Header do CSV
echo "dataset,np,N,K,max_iter,eps,iterations,ms_total,sse,points_per_sec" > "$OUT"

# --- 2. LOOP DOS DATASETS (10^4, 10^5, 10^6) ---
# Lista de pastas que voce tem em ../gerardados
SIZES=("104" "105" "106")

echo "--- 2. Iniciando Bateria de Testes ---"

for SZ in "${SIZES[@]}"; do
    # Define caminhos baseados na pasta
    DATA="../gerardados/$SZ/dados.csv"
    CINIT="../gerardados/$SZ/centroides_iniciais.csv"
    
    # Parametros padrao
    MAXIT=50
    EPS=1e-4

    echo "========================================"
    echo "PROCESSANDO DATASET: 10^$SZ"
    echo "Input: $DATA"

    # Verifica se os arquivos existem antes de tentar rodar
    if [ ! -f "$DATA" ]; then
        echo "AVISO: Arquivo $DATA nao encontrado. Pulando..."
        continue
    fi

    # Detecta N e K automaticamente (conta linhas nao vazias)
    N=$(grep -cve '^\s*$' "$DATA")
    K=$(grep -cve '^\s*$' "$CINIT")
    echo "Detectado: N=$N, K=$K"

    # --- 3. LOOP DE PROCESSOS (Strong Scaling) ---
    # Testa com 1, 2 e 4 processos
    PROCS_LIST="1 2 4"

    for NP in $PROCS_LIST; do
        echo "   -> Rodando com NP = $NP..."
        
        # Executa o MPI
        LOG=$(mpiexec -n "$NP" ./kmeans_mpi.exe "$DATA" "$CINIT" "$MAXIT" "$EPS" - -)
        
        # Parsing (Extrai os numeros do texto de saida)
        RES_LINE=$(echo "$LOG" | grep "RESULTADO")

        if [ -z "$RES_LINE" ]; then
            echo "      [ERRO] Sem saida valida do MPI."
            continue
        fi

        # Extrai valores usando awk e cut
        ITERS=$(echo "$RES_LINE" | awk -F'|' '{print $3}' | cut -d':' -f2 | tr -d ' ')
        SSE=$(echo "$RES_LINE"   | awk -F'|' '{print $4}' | cut -d':' -f2 | tr -d ' ')
        MS_TOTAL=$(echo "$RES_LINE" | awk -F'|' '{print $5}' | cut -d':' -f2 | cut -d' ' -f2)

        # Calcula Throughput (Pontos por segundo)
        # Evita divisao por zero se o tempo for muito pequeno
        PPS=$(awk -v n="$N" -v ms="$MS_TOTAL" 'BEGIN{ if(ms>0.001){printf "%.2f", n*1000.0/ms} else {print 0} }')

        # Salva no CSV (Correcao: Agora salvamos $K em vez de 0)
        # Formato: dataset,np,N,K,max_iter,eps,iterations,ms_total,sse,points_per_sec
        echo "10^$SZ,$NP,$N,$K,$MAXIT,$EPS,$ITERS,$MS_TOTAL,$SSE,$PPS" >> "$OUT"
        
        echo "      Concluido: Tempo=${MS_TOTAL}ms | SSE=$SSE"
    done
done

echo "========================================"
echo "Bateria finalizada! Resultados salvos em: $OUT"