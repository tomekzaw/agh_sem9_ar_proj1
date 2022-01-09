import numpy as np


def compare(zad_i, np_i, zad_j, np_j):
    name_i = f'zad{zad_i}_np{np_i}'
    name_j = f'zad{zad_j}_np{np_j}'
    accelerations_i = np.load(f'accelerations_{name_i}.npy')
    accelerations_j = np.load(f'accelerations_{name_j}.npy')
    ok_count = np.isclose(accelerations_i, accelerations_j, rtol=1e-12, atol=0).sum()
    all_count = accelerations_i.size
    looks_same = str(accelerations_i) == str(accelerations_j)
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
