/* kmeans_1d_mpi.c
   K-means 1D com MPI (Etapa 3).
   - Distribui N pontos entre P processos.
   - Centróides são globais e replicados em todos os processos.
   - Assignment e Update distribuídos.
   - Comunicação via MPI_Allreduce para somas parciais e contagens.
   - Rank 0 lê/escreve arquivos e coordena.

   Compilar: mpicc -O2 kmeans_1d_mpi.c -o kmeans_1d_mpi -lm
   Uso:      mpirun -np <P> ./kmeans_1d_mpi dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv|-] [centroids.csv|-] [--csv]
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

/* ---------- util CSV 1D: cada linha tem 1 número ---------- */
static int count_rows(const char *path){
    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); exit(1); }
    int rows=0; char line[8192];
    while(fgets(line,sizeof(line),f)){
        int only_ws=1;
        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(!only_ws) rows++;
    }
    fclose(f);
    return rows;
}

static double *read_csv_1col(const char *path, int *n_out){
    int R = count_rows(path);
    if(R<=0){ fprintf(stderr,"Arquivo vazio: %s\n", path); exit(1); }
    double *A = (double*)malloc((size_t)R * sizeof(double));
    if(!A){ fprintf(stderr,"Sem memoria para %d linhas\n", R); exit(1); }

    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); free(A); exit(1); }

    char line[8192];
    int r=0;
    while(fgets(line,sizeof(line),f)){
        int only_ws=1;
        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(only_ws) continue;

        const char *delim = ",; \t";
        char *tok = strtok(line, delim);
        if(!tok){ fprintf(stderr,"Linha %d sem valor em %s\n", r+1, path); free(A); fclose(f); exit(1); }
        A[r] = atof(tok);
        r++;
        if(r>R) break;
    }
    fclose(f);
    *n_out = R;
    return A;
}

static int should_skip(const char *path){
    return (!path || !path[0] || strcmp(path,"-")==0);
}

static void write_assign_csv(const char *path, const int *assign, int N){
    if(should_skip(path)) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int i=0;i<N;i++) fprintf(f, "%d\n", assign[i]);
    fclose(f);
}

static void write_centroids_csv(const char *path, const double *C, int K){
    if(should_skip(path)) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int c=0;c<K;c++) fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}

int main(int argc, char **argv){
    MPI_Init(&argc, &argv); // Inscreve o processo na computação MPI (cada processo executa este main) / Cria uma barreira implícita

    int rank, size;
    int ierr;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Identificador do processo na comunicação, retorna um valor entre 0 e size-1 em rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Número total de processos da comunicação

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;

    MPI_Get_processor_name(processor_name, &name_len); // Obtém o nome do nó onde o processo está rodando

    printf("Processo %d de %d iniciado no nó %s\n", rank, size, processor_name);

    if(rank == 0 && argc < 3){
        printf("Uso: mpirun -np <P> %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv|-] [centroids.csv|-] [--csv]\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Parâmetros globais */
    int N = 0, K = 0;
    int max_iter = 50;
    double eps = 1e-4;
    int csv = 0;
    char *outAssign = "assign.csv";
    char *outCentroid = "centroids.csv";

    double *X = NULL; /* Apenas no Rank 0 */
    double *C = NULL; /* Em todos os ranks */
    int *assign = NULL; /* Apenas no Rank 0 para escrita final */

    /* Rank 0 lê argumentos e dados */
    if(rank == 0){
        const char *pathX = argv[1];
        const char *pathC = argv[2];
        if(argc>3 && argv[3][0] != '-') max_iter = atoi(argv[3]);
        if(argc>4 && argv[4][0] != '-') eps = atof(argv[4]);
        if(argc>5) outAssign = argv[5];
        if(argc>6) outCentroid = argv[6];
        
        /* Mapear string vazia para padrão */
        if(!outAssign || !outAssign[0]) outAssign = "assign.csv";
        if(!outCentroid || !outCentroid[0]) outCentroid = "centroids.csv";

        for(int i=1;i<argc;i++){
            if(strcmp(argv[i],"--csv")==0) { csv = 1; break; }
        }

        X = read_csv_1col(pathX, &N);
        C = read_csv_1col(pathC, &K);
        assign = (int*)malloc((size_t)N * sizeof(int));
    }

    /* Broadcast de parâmetros escalares */
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Alocar C em todos os processos */
    if(rank != 0) C = (double*)malloc((size_t)K * sizeof(double));
    MPI_Bcast(C, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Precisamos de X[0] para o caso de cluster vazio (fallback) */
    double X0 = 0.0;
    if(rank == 0) X0 = X[0];
    MPI_Bcast(&X0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Distribuição de X (Scatterv) */
    int *sendcounts = NULL;
    int *displs = NULL;
    int local_N = N / size;
    int remainder = N % size;

    if(rank == 0){
        sendcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        int offset = 0;
        for(int i=0; i<size; i++){
            sendcounts[i] = (N / size) + (i < remainder ? 1 : 0);
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    /* Ajustar local_N para cada processo */
    int my_local_N = (N / size) + (rank < remainder ? 1 : 0);
    double *local_X = (double*)malloc(my_local_N * sizeof(double));
    int *local_assign = (int*)malloc(my_local_N * sizeof(int));

    MPI_Scatterv(X, sendcounts, displs, MPI_DOUBLE, 
                 local_X, my_local_N, MPI_DOUBLE, 
                 0, MPI_COMM_WORLD);

    /* Buffers para redução */
    double *local_sum = (double*)malloc(K * sizeof(double));
    int *local_cnt = (int*)malloc(K * sizeof(int));
    double *global_sum = (double*)malloc(K * sizeof(double));
    int *global_cnt = (int*)malloc(K * sizeof(int));

    double prev_sse = 1e300;
    double sse = 0.0;
    int it = 0;
    int mono_ok = 1; /* Não estamos verificando monotonicidade distribuída aqui para simplificar, mas o SSE global deve ser monotônico */

    double t0 = MPI_Wtime();

    for(it=0; it<max_iter; it++){
        double local_sse = 0.0;
        
        /* Zerar acumuladores locais */
        for(int c=0; c<K; c++){
            local_sum[c] = 0.0;
            local_cnt[c] = 0;
        }

        /* Assignment Step (Local) */
        for(int i=0; i<my_local_N; i++){
            int best = -1;
            double bestd = 1e300;
            for(int c=0; c<K; c++){
                double diff = local_X[i] - C[c];
                double d = diff*diff;
                if(d < bestd){ bestd = d; best = c; }
            }
            local_assign[i] = best;
            local_sse += bestd;
            
            local_sum[best] += local_X[i];
            local_cnt[best]++;
        }

        /* Redução Global */
        MPI_Allreduce(&local_sse, &sse, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_sum, global_sum, K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_cnt, global_cnt, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        /* Update Step (Replicado) */
        for(int c=0; c<K; c++){
            if(global_cnt[c] > 0){
                C[c] = global_sum[c] / global_cnt[c];
            } else {
                C[c] = X0; /* Fallback igual ao OpenMP */
            }
        }

        /* Checagem de convergência */
        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if(rel < eps){
            it++; /* Contar esta iteração */
            break;
        }
        prev_sse = sse;
    }

    double t1 = MPI_Wtime();
    double ms = 1000.0 * (t1 - t0);

    /* Gather Results (Gatherv) */
    MPI_Gatherv(local_assign, my_local_N, MPI_INT,
                assign, sendcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    if(rank == 0){
        if(csv){
            /* CSV: N,K,max_iter,eps,P,iterations,ms,sse */
            printf("%d,%d,%d,%.8g,%d,%d,%.6f,%.9f\n",
                   N, K, max_iter, eps, size,
                   it, ms, sse);
        } else {
            printf("K-means 1D (MPI)\n");
            printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);
            printf("Processos=%d\n", size);
            printf("Iterações: %d | SSE final: %.9f | Tempo: %.3f ms\n",
                   it, sse, ms);
        }

        write_assign_csv(outAssign, assign, N);
        write_centroids_csv(outCentroid, C, K);

        free(X);
        free(assign);
        free(sendcounts);
        free(displs);
    }

    free(C);
    free(local_X);
    free(local_assign);
    free(local_sum);
    free(local_cnt);
    free(global_sum);
    free(global_cnt);

    MPI_Finalize();
    return 0;
}
