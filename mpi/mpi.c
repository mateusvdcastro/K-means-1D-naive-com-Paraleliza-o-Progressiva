#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

// --- Funções Auxiliares de Leitura (Idênticas ao Naive) ---
static int count_rows(const char *path) {
    FILE *f = fopen(path, "r");
    if(!f) return -1;
    int rows = 0; char line[8192];
    while(fgets(line, sizeof(line), f)){
        int only_ws=1;
        for(char *p=line; *p; p++) {
            if(*p!=' ' && *p!= '\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(!only_ws) rows++;
    }
    fclose(f);
    return rows;
}

static double *read_csv_1col(const char *path, int *n_out){
    int R = count_rows(path);
    if(R<=0) return NULL;
    double *A = (double*)malloc((size_t)R * sizeof(double));
    FILE *f = fopen(path, "r");
    if(!f) { free(A); return NULL; }
    char line[8192];
    int r=0;
    while(fgets(line, sizeof(line), f)){
        int only_ws=1;
        for(char *p=line; *p; p++) {
            if(*p!=' ' && *p!= '\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(only_ws) continue;
        char *tok = strtok(line, ",; \t");
        if(tok) { A[r++] = atof(tok); }
        if(r >= R) break;
    }
    fclose(f);
    *n_out = r;
    return A;
}

static void write_assign_csv(const char *path, const int *assign, int N){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f) return;
    for(int i=0;i<N;i++) fprintf(f, "%d\n", assign[i]);
    fclose(f);
}

static void write_centroids_csv(const char *path, const double *C, int K){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f) return;
    for(int c=0;c<K;c++) fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}

// --- Main MPI ---
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(argc < 3){
        if(rank == 0) printf("Uso: mpirun -np <P> %s dados.csv centroides.csv [max_iter] [eps] [outAssign] [outCentroids]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc>3) ? atoi(argv[3]) : 50;
    double eps = (argc>4) ? atof(argv[4]) : 1e-4;
    const char *outAssign = (argc>5 && strcmp(argv[5], "-") != 0) ? argv[5] : NULL;
    const char *outCentroid = (argc>6 && strcmp(argv[6], "-") != 0) ? argv[6] : NULL;

    int N_global = 0, K = 0;
    double *X_global = NULL;
    double *C = NULL;

    // 1. Mestre lê os arquivos
    if (rank == 0) {
        X_global = read_csv_1col(pathX, &N_global);
        C = read_csv_1col(pathC, &K);
        if(!X_global || !C) {
            fprintf(stderr, "Erro na leitura (Rank 0).\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // 2. Broadcast de N e K
    MPI_Bcast(&N_global, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Alocação de C nos workers
    if(rank != 0) {
        C = (double*)malloc(K * sizeof(double));
    }
    // Centróides iniciais iguais para todos
    MPI_Bcast(C, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 3. Preparar Scatterv (divisão de dados)
    int *sendcounts = NULL;
    int *displs = NULL;
    int N_local = N_global / size;
    int remainder = N_global % size;

    // Se rank < resto, ele pega um a mais
    if (rank < remainder) N_local++;

    if (rank == 0) {
        sendcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        int offset = 0;
        for (int i = 0; i < size; i++) {
            sendcounts[i] = (N_global / size) + (i < remainder ? 1 : 0);
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    double *X_local = (double*)malloc(N_local * sizeof(double));
    int *assign_local = (int*)malloc(N_local * sizeof(int));

    // Distribui os dados
    MPI_Scatterv(X_global, sendcounts, displs, MPI_DOUBLE,
                 X_local, N_local, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // --- Loop K-Means ---
    double *sum_local = (double*)malloc(K * sizeof(double));
    int *cnt_local = (int*)malloc(K * sizeof(int));
    double *sum_global = (double*)malloc(K * sizeof(double));
    int *cnt_global = (int*)malloc(K * sizeof(int));

    double prev_sse = 1e300;
    double sse_global = 0.0;
    int it = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    for(it = 0; it < max_iter; it++) {
        double sse_local = 0.0;
        for(int c=0; c<K; c++) { sum_local[c]=0.0; cnt_local[c]=0; }

        // Passo A: Assignment Local
        for(int i=0; i<N_local; i++) {
            double best_d = 1e300;
            int best_c = -1;
            for(int c=0; c<K; c++) {
                double d = (X_local[i] - C[c]) * (X_local[i] - C[c]);
                if(d < best_d) { best_d = d; best_c = c; }
            }
            assign_local[i] = best_c;
            sse_local += best_d;
            sum_local[best_c] += X_local[i];
            cnt_local[best_c]++;
        }

        // Passo B: Redução Global (Somas e Contagens)
        MPI_Allreduce(&sse_local, &sse_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(sum_local, sum_global, K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(cnt_local, cnt_global, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // Passo C: Convergência
        double rel = fabs(sse_global - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if(rel < eps) {
            it++;
            break;
        }
        prev_sse = sse_global;

        // Passo D: Atualizar Centróides
        for(int c=0; c<K; c++) {
            if(cnt_global[c] > 0) C[c] = sum_global[c] / cnt_global[c];
            // Se vazio, mantém o anterior (naive)
        }
    }

    double end_time = MPI_Wtime();

    // 4. Coletar resultados (opcional, só se tiver output)
    int *assign_global = NULL;
    if (outAssign && rank == 0) {
        assign_global = (int*)malloc(N_global * sizeof(int));
    }
    if (outAssign) {
        MPI_Gatherv(assign_local, N_local, MPI_INT,
                    assign_global, sendcounts, displs, MPI_INT,
                    0, MPI_COMM_WORLD);
    }

    // 5. Output Final (Rank 0)
    if(rank == 0) {
        double total_ms = (end_time - start_time) * 1000.0;
        // Formato para facilitar o parsing do script
        printf("RESULTADO | Np: %d | Iterações: %d | SSE final: %.6f | Tempo Total: %.3f ms\n",
               size, it, sse_global, total_ms);

        if(outAssign) {
            write_assign_csv(outAssign, assign_global, N_global);
            free(assign_global);
        }
        if(outCentroid) write_centroids_csv(outCentroid, C, K);

        free(X_global);
        free(sendcounts);
        free(displs);
    }

    free(X_local);
    free(assign_local);
    free(sum_local);
    free(cnt_local);
    free(sum_global);
    free(cnt_global);
    free(C);

    MPI_Finalize();
    return 0;
}