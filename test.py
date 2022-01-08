import numpy as np


def compare(zad_i, np_i, zad_j, np_j):
    name_i = f'zad{zad_i}_np{np_i}'
    name_j = f'zad{zad_j}_np{np_j}'
    interactions_i = np.load(f'interactions_{name_i}.npy')
    interactions_j = np.load(f'interactions_{name_j}.npy')
    ok_count = np.isclose(interactions_i, interactions_j, rtol=1e-12, atol=0).sum()
    all_count = interactions_i.size
    looks_same = str(interactions_i) == str(interactions_j)
    print(f'{name_i} vs {name_j}: {ok_count}/{all_count} {looks_same}')


if __name__ == '__main__':
    zads = [1, 2]
    nps = [2, 3, 4, 5, 6, 7, 8]

    for zad in zads:
        for j in nps:
            compare(zad, 1, zad, j)
        print()

    compare(1, 1, 2, 1)
    for j in nps:
        compare(1, j, 2, j)
    print()
