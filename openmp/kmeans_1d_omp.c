/* kmeans_1d_omp.c
   K-means 1D (C99) com OpenMP (Etapa 1).
   - Paraleliza:
       * Assignment: laço for (i=0; i<N; ++i) com reduction em SSE.
       * Update: acumuladores por thread (sum_thread[c], cnt_thread[c]) e redução após a região paralela.
   - Mede tempo com omp_get_wtime().
   - Suporta ajuste de schedule (static|dynamic|guided|auto) e chunk via omp_set_schedule() + schedule(runtime).
   - Saída humana padrão ou CSV (--csv) para uso em scripts.
   - **Agora**: por padrão SEMPRE escreve assign.csv e centroids.csv, a não ser que você passe "-" para pular.

   Compilar: gcc -O2 -fopenmp -std=c99 kmeans_1d_omp.c -o kmeans_1d_omp -lm
   Uso:      ./kmeans_1d_omp dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv|-] [centroids.csv|-] [schedule=static] [chunk=0] [--csv]

   Observações:
   - Controle de threads: export OMP_NUM_THREADS=T (ou use omp_set_num_threads).
   - "chunk=0" deixa a runtime decidir.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

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

/* ---------- ASSIGNMENT (OpenMP) ---------- */
static double assignment_step_1d_omp(const double *X, const double *C, int *assign, int N, int K){
    double sse = 0.0;
    #pragma omp parallel for reduction(+:sse) schedule(runtime)
    for(int i=0;i<N;i++){
        int best = -1;
        double bestd = 1e300;
        for(int c=0;c<K;c++){
            double diff = X[i] - C[c];
            double d = diff*diff;
            if(d < bestd){ bestd = d; best = c; }
        }
        assign[i] = best;
        sse += bestd;
    }
    return sse;
}

/* ---------- UPDATE (OpenMP, opção A) ---------- */
static void update_step_1d_omp(const double *X, double *C, const int *assign, int N, int K){
    int T = omp_get_max_threads();
    double *sumt = (double*)calloc((size_t)T * (size_t)K, sizeof(double));
    int    *cntt = (int*)   calloc((size_t)T * (size_t)K, sizeof(int));
    if(!sumt || !cntt){ fprintf(stderr,"Sem memoria no update\n"); exit(1); }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double *sum_local = sumt + (size_t)tid*(size_t)K;
        int    *cnt_local = cntt + (size_t)tid*(size_t)K;

        #pragma omp for schedule(runtime)
        for(int i=0;i<N;i++){
            int a = assign[i];
            cnt_local[a] += 1;
            sum_local[a] += X[i];
        }
    }

    for(int c=0;c<K;c++){
        double sum = 0.0;
        int cnt = 0;
        for(int t=0;t<T;t++){
            sum += sumt[t*(size_t)K + c];
            cnt += cntt[t*(size_t)K + c];
        }
        if(cnt > 0) C[c] = sum / (double)cnt; else C[c] = X[0];
    }

    free(sumt); free(cntt);
}

static void set_schedule_from_args(const char *sched_str, int chunk){
    omp_sched_t policy = omp_sched_static;
    if(sched_str){
        if(strcmp(sched_str,"static")==0) policy = omp_sched_static;
        else if(strcmp(sched_str,"dynamic")==0) policy = omp_sched_dynamic;
        else if(strcmp(sched_str,"guided")==0)  policy = omp_sched_guided;
        else if(strcmp(sched_str,"auto")==0)    policy = omp_sched_auto;
        else { fprintf(stderr,"[Aviso] schedule desconhecido '%s', usando static.\n", sched_str); }
    }
    omp_set_schedule(policy, chunk);
}

static void get_schedule_str(char *out, size_t cap, int *chunk_out){
    omp_sched_t policy; int chunk;
    omp_get_schedule(&policy, &chunk);
    const char *name = "static";
    if(policy==omp_sched_dynamic) name="dynamic";
    else if(policy==omp_sched_guided) name="guided";
    else if(policy==omp_sched_auto) name="auto";
    snprintf(out, cap, "%s", name);
    if(chunk_out) *chunk_out = chunk;
}

static int is_monotonic_nonincreasing(const double *sse_hist, int iters){
    for(int i=1;i<iters;i++){
        if(sse_hist[i] > sse_hist[i-1] + 1e-9) return 0;
    }
    return 1;
}

static void kmeans_1d_omp(const double *X, double *C, int *assign,
                          int N, int K, int max_iter, double eps,
                          int *iters_out, double *sse_out, int *mono_ok, double *sse_hist_out)
{
    double prev_sse = 1e300;
    double sse = 0.0;
    int it;
    for(it=0; it<max_iter; it++){
        sse = assignment_step_1d_omp(X, C, assign, N, K);
        sse_hist_out[it] = sse;
        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if(rel < eps){ it++; break; }
        update_step_1d_omp(X, C, assign, N, K);
        prev_sse = sse;
    }
    *iters_out = it;
    *sse_out = sse;
    *mono_ok = is_monotonic_nonincreasing(sse_hist_out, it);
}

int main(int argc, char **argv){
    if(argc < 3){
        printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv|-] [centroids.csv|-] [schedule=static] [chunk=0] [--csv]\n", argv[0]);
        printf("Obs: arquivos CSV com 1 coluna (1 valor por linha), sem cabeçalho.\n");
        return 1;
    }
    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc>3 && argv[3][0] != '-')? atoi(argv[3]) : 50;
    double eps   = (argc>4 && argv[4][0] != '-')? atof(argv[4]) : 1e-4;

    /* Por padrão, vamos escrever assign.csv e centroids.csv;
       Se o usuário passar "-", pulamos a escrita. Se passar string vazia, usamos o padrão. */
    const char *outAssign   = (argc>5)? argv[5] : "assign.csv";
    const char *outCentroid = (argc>6)? argv[6] : "centroids.csv";

    /* Mapear string vazia para padrão */
    if(!outAssign || !outAssign[0]) outAssign = "assign.csv";
    if(!outCentroid || !outCentroid[0]) outCentroid = "centroids.csv";

    const char *sched_str   = (argc>7 && argv[7][0] != '-')? argv[7] : "static";
    int chunk               = (argc>8 && argv[8][0] != '-')? atoi(argv[8]) : 0;

    int csv = 0;
    for(int i=1;i<argc;i++){
        if(strcmp(argv[i],"--csv")==0) { csv = 1; break; }
    }

    set_schedule_from_args(sched_str, chunk);

    int N=0, K=0;
    double *X = read_csv_1col(pathX, &N);
    double *C = read_csv_1col(pathC, &K);
    int *assign = (int*)malloc((size_t)N * sizeof(int));
    if(!assign){ fprintf(stderr,"Sem memoria para assign\n"); free(X); free(C); return 1; }

    double *sse_hist = (double*)malloc((size_t) (max_iter>0?max_iter:1) * sizeof(double));
    if(!sse_hist){ fprintf(stderr,"Sem memoria para sse_hist\n"); free(assign); free(X); free(C); return 1; }

    double t0 = omp_get_wtime();
    int iters = 0; double sse = 0.0; int mono_ok = 1;
    kmeans_1d_omp(X, C, assign, N, K, max_iter, eps, &iters, &sse, &mono_ok, sse_hist);
    double t1 = omp_get_wtime();
    double ms = 1000.0 * (t1 - t0);

    char sched_name[32]; int used_chunk=0;
    get_schedule_str(sched_name, sizeof(sched_name), &used_chunk);

    if(csv){
        /* CSV: N,K,max_iter,eps,T_used,schedule,chunk,iterations,ms,sse,monotonic */
        printf("%d,%d,%d,%.8g,%d,%s,%d,%d,%.6f,%.9f,%d\n",
               N, K, max_iter, eps, omp_get_max_threads(), sched_name, used_chunk,
               iters, ms, sse, mono_ok);
    } else {
        printf("K-means 1D (OpenMP)\n");
        printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);
        printf("threads=%d schedule=%s chunk=%d\n", omp_get_max_threads(), sched_name, used_chunk);
        printf("Iterações: %d | SSE final: %.9f | Tempo: %.3f ms | Monotônico: %s\n",
               iters, sse, ms, mono_ok? "sim":"NAO");
    }

    /* Escrever arquivos de saída (pula se path == "-") */
    write_assign_csv(outAssign, assign, N);
    write_centroids_csv(outCentroid, C, K);

    free(sse_hist);
    free(assign); free(X); free(C);
    return 0;
}
