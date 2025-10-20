# K-means-1D-naive-com-Paraleliza-o-Progressiva

gcc -O2 -std=c99 kmeans_1d_naive.c -o kmeans_1d_naive -lm

./kmeans_1d_naive dados.csv centroides_iniciais.csv 50 0.000001 assign.csv centroids.csv

cat centroids.csv