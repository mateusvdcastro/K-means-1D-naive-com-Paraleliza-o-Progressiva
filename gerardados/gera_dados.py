#!/usr/bin/env python3
import numpy as np, sys
import numpy as np, matplotlib.pyplot as plt

def gen(N, K, centers=None, sigma=1.0, seed=123, uniform=False):
    rng = np.random.default_rng(seed)
    if centers is None:
        # centros igualmente espaçados de 10 em 10: 0,10,20,...
        centers = np.arange(K, dtype=float) * 10.0
    # divide N quase uniformemente entre os K grupos
    counts = np.full(K, N//K, dtype=int)
    counts[:N - counts.sum()] += 1

    parts = []
    for c, cnt in zip(centers, counts):
        if uniform:
            parts.append(rng.uniform(c-3*sigma, c+3*sigma, size=cnt))
        else:
            parts.append(rng.normal(c, sigma, size=cnt))
    X = np.concatenate(parts)
    rng.shuffle(X)

    # centróides iniciais simples: K pontos igualmente espaçados no range observado
    xmin, xmax = float(X.min()), float(X.max())
    C = xmin + (np.arange(K)+0.5)*(xmax - xmin)/K
    return X, C

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("uso: python gera_dados.py N K [sigma=1.0] [seed=123]")
        sys.exit(1)
    N = int(sys.argv[1]); K = int(sys.argv[2])
    sigma = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    seed  = int(sys.argv[4]) if len(sys.argv) > 4 else 123
    X, C = gen(N, K, sigma=sigma, seed=seed, uniform=False)
    np.savetxt("dados.csv", X, fmt="%.6f")
    np.savetxt("centroides_iniciais.csv", C, fmt="%.6f")
    print(f"Gerado: dados.csv (N={N}), centroides_iniciais.csv (K={K}), sigma={sigma}, seed={seed}")