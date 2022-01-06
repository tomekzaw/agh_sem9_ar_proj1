#!/usr/bin/env python

import numpy as np

from mpi4py import MPI

G = 6.6743015e-11  # from https://en.wikipedia.org/wiki/Gravitational_constant

N = 400  # liczba wszystkich gwiazd


def calculate_interactions_own(stars_own: np.array) -> np.array:
    n = stars_own.shape[0]
    interactions = np.zeros((n, 3))
    return interactions


def calculate_interactions_own_with_buffer(stars_own: np.array, stars_buffer: np.array) -> np.array:
    n = stars_own.shape[0]
    interactions = np.zeros((n, 3))
    return interactions


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    size = p = comm.Get_size()
    rank = comm.Get_rank()

    left = (rank - 1) % size
    right = (rank + 1) % size

    # Każdy z procesów zawiera równoliczny zbiór gwiazd N/p, ...
    stars_own = np.random.random((N // p, 4))  # M, x, y, z; TODO: vx, vy, vz

    # ...bufor na gwiazdy przechodnie...
    # stars_buffer # M, x, y, z, vx, vy, vz

    # ...oraz akumulator swoich interakcji.
    accumulator = calculate_interactions_own(stars_own)  # ax, ay, az

    # Początkowo w buforze przechodnim znajdują się własne gwazdy danego procesu.
    stars_buffer = stars_own.copy()

    # Każdy z procesów powtarza p-1 razy następujące czynności:
    for _ in range(p - 1):

        # 1. przesyła przechodnią porcję(N/p) gwiazd do lewego sąsiada w pierścieniu procesów
        comm.Isend([stars_buffer, MPI.FLOAT], dest=left)

        # 2. odbiera porcję (N/p) gwiazd od prawego sąsiada do swojego bufora przechodniego
        comm.Recv([stars_buffer, MPI.FLOAT], source=right)

        # 3. oblicza interakcje swoich gwiazd i gwiazd otrzymanych
        interactions = calculate_interactions_own_with_buffer(
            stars_own, stars_buffer)

        # 4. dodaje obliczona interakcje do swojego akumulatora
        accumulator += interactions

    print('Finished!')
