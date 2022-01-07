import numpy as np

if __name__ == '__main__':
    nproc = 4
    for p in range(2, nproc+1):
        interactions_1 = np.load(f'interactions_1.npy')
        interactions_p = np.load(f'interactions_{p}.npy')
        ok = np.isclose(interactions_1, interactions_p).sum()
        print(f'1-{p}: {ok}/{interactions_1.size}')
