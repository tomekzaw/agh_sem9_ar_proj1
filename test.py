import numpy as np
from itertools import combinations

if __name__ == '__main__':
    nproc = 4
    for i, j in combinations(range(1, nproc+1), 2):
        interactions_i = np.load(f'interactions_{i}.npy')
        interactions_j = np.load(f'interactions_{j}.npy')
        ok = np.isclose(interactions_i, interactions_j, rtol=1e-12, atol=0).sum()
        print(f'{i}-{j}: {ok}/{interactions_i.size}')
