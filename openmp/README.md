# K-means-1D-naive-com-Paraleliza-o-Progressiva

1) gcc -O2 -fopenmp -std=c99 kmeans_1d_omp.c -o kmeans_1d_omp -lm

2) export OMP_NUM_THREADS=4

3) ./kmeans_1d_omp ../gerardados/104/dados.csv ../gerardados/104/centroides_iniciais.csv 50 1e-4 assign104.csv centroids104.csv dynamic 1 --csv

./run_omp_sweep.sh ../gerardados/104/dados.csv ../gerardados/104/centroides_iniciais.csv 50 1e-4 results_104.csv

mv assign104.csv centroids104.csv results_104.csv ../gerardados/104/results/

4) ./kmeans_1d_omp ../gerardados/105/dados.csv ../gerardados/105/centroides_iniciais.csv 50 1e-4 assign105.csv centroids105.csv dynamic 1 --csv

./run_omp_sweep.sh ../gerardados/105/dados.csv ../gerardados/105/centroides_iniciais.csv 50 1e-4 results_105.csv

mv assign105.csv centroids105.csv results_105.csv ../gerardados/105/results/

5) ./kmeans_1d_omp ../gerardados/106/dados.csv ../gerardados/106/centroides_iniciais.csv 50 1e-4 assign106.csv centroids106.csv dynamic 1 --csv

./run_omp_sweep.sh ../gerardados/106/dados.csv ../gerardados/106/centroides_iniciais.csv 50 1e-4 results_106.csv

mv assign106.csv centroids106.csv results_106.csv ../gerardados/106/results/
