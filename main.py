#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
from timeit import default_timer as timer
from utils import calculate_interactions_own, calculate_interactions_own_with_buffer

N = 1200  # liczba wszystkich gwiazd

if __name__ == '__main__':
    start_time = timer()

    comm = MPI.COMM_WORLD
    p = comm.Get_size()
    rank = comm.Get_rank()

    assert N % p == 0  # TODO: support number of stars indivisible by number of processes
    n = N // p

    left = (rank - 1) % p
    right = (rank + 1) % p

    # Każdy z procesów zawiera równoliczny zbiór gwiazd N/p, ...
    # stars_own = np.random.random((n, 4))  # M, x, y, z
    start = rank * n
    end = (rank + 1) * n
    stars_own = np.load('stars.npy')[start:end, :]

    # ...bufor na gwiazdy przechodnie...
    # stars_buffer # M, x, y, z
    # ...oraz akumulator swoich interakcji.
    accumulator = calculate_interactions_own(stars_own)  # ax, ay, az

    # Początkowo w buforze przechodnim znajdują się własne gwazdy danego procesu.
    stars_buffer = np.concatenate([stars_own] * 2)  # double buffering
    stars_buffer_A = stars_buffer[:n]
    stars_buffer_B = stars_buffer[n:]

    # Każdy z procesów powtarza p-1 razy następujące czynności:
    for _ in range(p-1):

        # 1. przesyła przechodnią porcję(N/p) gwiazd do lewego sąsiada w pierścieniu procesów
        comm.Isend([stars_buffer_A, MPI.FLOAT], dest=left)

        # 2. odbiera porcję (N/p) gwiazd od prawego sąsiada do swojego bufora przechodniego
        comm.Recv([stars_buffer_B, MPI.FLOAT], source=right)

        # 3. oblicza interakcje swoich gwiazd i gwiazd otrzymanych
        interactions = calculate_interactions_own_with_buffer(stars_own, stars_buffer_B)

        # 4. dodaje obliczona interakcje do swojego akumulatora
        accumulator += interactions

        stars_buffer_A, stars_buffer_B = stars_buffer_B, stars_buffer_A

    # zebranie danych
    interactions = comm.gather(accumulator)
    if rank == 0:  # root
        interactions = np.concatenate(interactions)

    duration = timer() - start_time

    if rank == 0:  # root
        # print(interactions)
        print(interactions.shape, p, duration)
        np.save(f'interactions_{p}.npy', interactions)
