#!/usr/bin/env python

import time
import numpy as np
from mpi4py import MPI
from utils import calculate_interactions_own, calculate_interactions_own_with_buffer

N = 400  # liczba wszystkich gwiazd

if __name__ == '__main__':
    # rozpoczęcie pomiaru czasu
    start_time = time.time()

    comm = MPI.COMM_WORLD
    size = p = comm.Get_size()
    rank = comm.Get_rank()

    assert N % p == 0

    left = (rank - 1) % size
    right = (rank + 1) % size

    # Każdy z procesów zawiera równoliczny zbiór gwiazd N/p, ...
    # stars_own = np.random.random((N // p, 4))  # M, x, y, z; TODO: vx, vy, vz
    start = rank * (N // p)
    end = (rank + 1) * (N // p)
    stars_own = np.load('stars.npy')[start:end, :]

    # ...bufor na gwiazdy przechodnie...
    # stars_buffer # M, x, y, z, vx, vy, vz

    # ...oraz akumulator swoich interakcji.
    accumulator = calculate_interactions_own(stars_own)  # ax, ay, az
    # accumulator = calculate_interactions_own_with_buffer(stars_own, stars_own)

    # Początkowo w buforze przechodnim znajdują się własne gwazdy danego procesu.
    stars_buffer = stars_own.copy()

    # Każdy z procesów powtarza p-1 razy następujące czynności:
    for k in range(p - 1):

        # 1. przesyła przechodnią porcję(N/p) gwiazd do lewego sąsiada w pierścieniu procesów
        # comm.Send([stars_buffer, MPI.FLOAT], dest=left, tag=k)
        comm.isend(stars_buffer.copy(), dest=left, tag=k)

        # 2. odbiera porcję (N/p) gwiazd od prawego sąsiada do swojego bufora przechodniego
        # comm.Recv([stars_buffer, MPI.FLOAT], source=right, tag=k)
        stars_buffer = comm.recv(source=right, tag=k)

        # 3. oblicza interakcje swoich gwiazd i gwiazd otrzymanych
        interactions = calculate_interactions_own_with_buffer(
            stars_own, stars_buffer)

        # 4. dodaje obliczona interakcje do swojego akumulatora
        accumulator += interactions

    # zebranie danych
    interactions = comm.gather(accumulator)
    if not rank:
        interactions = np.concatenate(interactions)

    # koniec pomiaru czasu
    end_time = time.time()
    duration = end_time - start_time

    checksum = comm.reduce(accumulator.sum(), op=MPI.SUM, root=0)
    if rank == 0:
        print(interactions.shape, checksum, duration)
