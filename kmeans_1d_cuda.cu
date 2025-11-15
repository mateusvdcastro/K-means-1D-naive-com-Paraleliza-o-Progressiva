#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <float.h> // Necessário para FLT_MAX (o "infinito" de float)

inline void gpuAssert(cudaError_t code, const char *file, int line)
{
  if (code != cudaSuccess)
  {
   fprintf(stderr,"ERRO GPU: %s no arquivo %s linha %d\n", cudaGetErrorString(code), file, line);
   exit(code);
  }
}

#define gpuErrchk(ans) \
  do { \
    gpuAssert((ans), __FILE__, __LINE__); \
  } while (0)

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

static float *read_csv_1col(const char *path, int *n_out){
  int R = count_rows(path);
  if(R<=0){ fprintf(stderr,"Arquivo vazio: %s\n", path); exit(1); }
  float *A = (float*)malloc((size_t)R * sizeof(float));
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
    A[r] = (float)atof(tok);
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

static void write_centroids_csv(const char *path, const float *C, int K){
  if(should_skip(path)) return;
  FILE *f = fopen(path, "w");
  if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
  for(int c=0;c<K;c++) fprintf(f, "%.6f\n", C[c]);
  fclose(f);
}

__global__
void assignment_kernel_1d(const float *X, const float *C, int *assign, float *sse_errors, int N, int K) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i == 0) {
    // Este printf não acessa ponteiros de dados, apenas N e K
    // printf(">>> KERNEL: Thread 0 está VIVA! (N=%d, K=%d)\n", N, K); // Comentado para limpar o log
  }

  if (i < N) {
    float bestd = FLT_MAX;
    int best = -1;
    for (int c = 0; c < K; c++) {
      float diff = X[i] - C[c];
      float d = diff * diff;
      if (d < bestd) {
        bestd = d;
        best = c;
      }
    }
    assign[i] = best;
    sse_errors[i] = bestd;
  }
}

static void update_step_1d_host(const float *X, float *C, const int *assign, int N, int K){
  float *sum = (float*)calloc((size_t)K, sizeof(float));
  int *cnt = (int*)calloc((size_t)K, sizeof(int));
  if(!sum || !cnt) { fprintf(stderr,"Sem memoria no update (host)\n"); exit(1); }

  for(int i=0;i<N;i++){
    int a = assign[i];
    if (a < 0 || a >= K) continue;
    cnt[a] += 1;
    sum[a] += X[i];
  }
  for(int c=0;c<K;c++){
    if(cnt[c] > 0) C[c] = sum[c] / (float)cnt[c];
    else C[c] = X[0];
  }
  free(sum); free(cnt);
}

int main(int argc, char **argv){
  if(argc < 3){
    printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv|-] [centroids.csv|-] [blockSize=256]\n", argv[0]);
    printf("Obs: arquivos CSV com 1 coluna (1 valor por linha), sem cabeçalho.\n");
    return 1;
  }
  const char *pathX = argv[1];
  const char *pathC = argv[2];
  int max_iter = (argc>3 && argv[3][0] != '-')? atoi(argv[3]) : 50;
  float eps  = (argc>4 && argv[4][0] != '-')? (float)atof(argv[4]) : 1e-4f;
  const char *outAssign  = (argc>5)? argv[5] : "assign.csv";
  const char *outCentroid = (argc>6)? argv[6] : "centroids.csv";
  // *** MODIFICAÇÃO: Lê o blockSize do argumento 7 ***
  int blockSize = (argc>7)? atoi(argv[7]) : 256;

  if(!outAssign || !outAssign[0]) outAssign = "assign.csv";
  if(!outCentroid || !outCentroid[0]) outCentroid = "centroids.csv";

  // --- 1. Leitura de dados (HOST) ---
  int N=0, K=0;
  float *h_X = read_csv_1col(pathX, &N);
  float *h_C = read_csv_1col(pathC, &K);

  int *h_assign = (int*)malloc((size_t)N * sizeof(int));
  float *h_sse_errors = (float*)malloc((size_t)N * sizeof(float));
  float *sse_hist = (float*)malloc((size_t) (max_iter>0?max_iter:1) * sizeof(float));

  if(!h_assign || !h_sse_errors || !sse_hist){
    fprintf(stderr,"Sem memoria para arrays de host\n");
    free(h_X); free(h_C); return 1;
  }

  // --- 2. Alocação de memória (DEVICE/GPU) ---
  float *d_X, *d_C, *d_sse_errors;
  int *d_assign;

  gpuErrchk(cudaMalloc(&d_X, (size_t)N * sizeof(float)));
  gpuErrchk(cudaMalloc(&d_C, (size_t)K * sizeof(float)));
  gpuErrchk(cudaMalloc(&d_assign, (size_t)N * sizeof(int)));
  gpuErrchk(cudaMalloc(&d_sse_errors, (size_t)N * sizeof(float)));


  // --- 4. Medição de Tempo (CUDA Events) ---
  float ms_total = 0; // Tempo total do loop
  cudaEvent_t start, stop;
  gpuErrchk(cudaEventCreate(&start));
  gpuErrchk(cudaEventCreate(&stop));

  // *** MODIFICAÇÃO: Eventos e acumuladores para medições granulares ***
  cudaEvent_t iter_start, iter_stop;
  gpuErrchk(cudaEventCreate(&iter_start));
  gpuErrchk(cudaEventCreate(&iter_stop));
  float iter_ms = 0.0f; // Tempo temporário por evento
  float ms_init_h2d = 0.0f; // Tempo da cópia H2D inicial
  float acc_ms_kernel = 0.0f; // Acumulador: Tempo de Kernel
  float acc_ms_h2d = 0.0f;  // Acumulador: Tempo de H2D (dentro do loop)
  float acc_ms_d2h = 0.0f;  // Acumulador: Tempo de D2H (dentro do loop)


  // --- 3. Cópia Inicial (Host -> Device) ---
  // *** MODIFICAÇÃO: Medir tempo da cópia inicial ***
  gpuErrchk(cudaEventRecord(iter_start));
  gpuErrchk(cudaMemcpy(d_X, h_X, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_C, h_C, (size_t)K * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaEventRecord(iter_stop));
  gpuErrchk(cudaEventSynchronize(iter_stop));
  gpuErrchk(cudaEventElapsedTime(&ms_init_h2d, iter_start, iter_stop));


  // --- Marca o início do tempo TOTAL ---
  gpuErrchk(cudaEventRecord(start));

  // --- 5. Loop de Iteração (Orquestrado pelo Host) ---
  float prev_sse = FLT_MAX;
  float sse = 0.0f;
  int iters = 0;
  int mono_ok = 1;

  for(iters=0; iters<max_iter; iters++){

    // --- 5a. Lançar Kernel (Assignment na GPU) ---
    // *** MODIFICAÇÃO: Usa o 'blockSize' vindo do argv ***
    int gridSize = (N + blockSize - 1) / blockSize;

    if (iters == 0) {
      // *** MODIFICAÇÃO: Imprime o gridSize para o script de sweep ***
      printf("GRID: %d\n", gridSize);
      printf("HOST (iter 0): Lançando kernel com N=%d, K=%d, gridSize=%d, blockSize=%d\n", N, K, gridSize, blockSize);
    }

    // *** MODIFICAÇÃO: Medir tempo de KERNEL ***
    gpuErrchk(cudaEventRecord(iter_start));
    assignment_kernel_1d<<<gridSize, blockSize>>>(d_X, d_C, d_assign, d_sse_errors, N, K);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaEventRecord(iter_stop));
    gpuErrchk(cudaEventSynchronize(iter_stop));
    gpuErrchk(cudaEventElapsedTime(&iter_ms, iter_start, iter_stop));
    acc_ms_kernel += iter_ms;


    // --- 5b. Copiar Resultados (Device -> Host) ---
    // *** MODIFICAÇÃO: Medir tempo de D2H (sse_errors) ***
    gpuErrchk(cudaEventRecord(iter_start));
    gpuErrchk(cudaMemcpy(h_sse_errors, d_sse_errors, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaEventRecord(iter_stop));
    gpuErrchk(cudaEventSynchronize(iter_stop));
    gpuErrchk(cudaEventElapsedTime(&iter_ms, iter_start, iter_stop));
    acc_ms_d2h += iter_ms;

    // --- 5c. Redução do SSE (na CPU) ---
    sse = 0.0f;
    for(int i=0; i<N; i++) {
      sse += h_sse_errors[i];
    }
    sse_hist[iters] = sse;

    // --- 5d. Checar Convergência (na CPU) ---
    float rel = fabsf(sse - prev_sse) / (prev_sse > 0.0f ? prev_sse : 1.0f);
    if(rel < eps){
      if(iters > 0 && sse > prev_sse + 1e-6f) mono_ok = 0;
      iters++;
      break;
    }
    if(iters > 0 && sse > prev_sse + 1e-6f) mono_ok = 0;
    prev_sse = sse;

    // --- 5e. Update (na CPU - Opção A) ---
    // *** MODIFICAÇÃO: Medir tempo de D2H (assign) ***
    gpuErrchk(cudaEventRecord(iter_start));
    gpuErrchk(cudaMemcpy(h_assign, d_assign, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaEventRecord(iter_stop));
    gpuErrchk(cudaEventSynchronize(iter_stop));
    gpuErrchk(cudaEventElapsedTime(&iter_ms, iter_start, iter_stop));
    acc_ms_d2h += iter_ms;

    update_step_1d_host(h_X, h_C, h_assign, N, K);

    // --- 5f. Enviar Centróides (Host -> Device) ---
    // *** MODIFICAÇÃO: Medir tempo de H2D (centroides) ***
    gpuErrchk(cudaEventRecord(iter_start));
    gpuErrchk(cudaMemcpy(d_C, h_C, (size_t)K * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaEventRecord(iter_stop));
    gpuErrchk(cudaEventSynchronize(iter_stop));
    gpuErrchk(cudaEventElapsedTime(&iter_ms, iter_start, iter_stop));
    acc_ms_h2d += iter_ms;
  }

  // --- 6. Fim da Medição ---
  gpuErrchk(cudaEventRecord(stop));
  gpuErrchk(cudaEventSynchronize(stop));
  gpuErrchk(cudaEventElapsedTime(&ms_total, start, stop));

  // --- 7. Saída e Escrita de Arquivos ---
  // (A última cópia de h_assign não precisa ser medida, já está fora do loop principal)
  gpuErrchk(cudaMemcpy(h_assign, d_assign, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost));

  printf("K-means 1D (CUDA)\n");
  printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);

  // *** MODIFICAÇÃO: Novo printf com todas as métricas de tempo ***
  printf("Iterações: %d | SSE final: %.9f | Tempo Total: %.3f ms | T. Kernel: %.3f ms | T. H2D: %.3f ms | T. D2H: %.3f ms | Monotônico: %s\n",
     iters, sse, ms_total, acc_ms_kernel, (ms_init_h2d + acc_ms_h2d), acc_ms_d2h, mono_ok? "sim":"NAO");

  write_assign_csv(outAssign, h_assign, N);
  write_centroids_csv(outCentroid, h_C, K);

  // --- 8. Limpeza (Host e Device) ---
  free(sse_hist);
  free(h_assign);
  free(h_sse_errors);
  free(h_X);
  free(h_C);

  gpuErrchk(cudaFree(d_X));
  gpuErrchk(cudaFree(d_C));
  gpuErrchk(cudaFree(d_assign));
  gpuErrchk(cudaFree(d_sse_errors));

  gpuErrchk(cudaEventDestroy(start));
  gpuErrchk(cudaEventDestroy(stop));
  // *** MODIFICAÇÃO: Limpeza dos novos eventos ***
  gpuErrchk(cudaEventDestroy(iter_start));
  gpuErrchk(cudaEventDestroy(iter_stop));

  return 0;
}