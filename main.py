#!/usr/bin/env python

import argparse
import numpy as np

from mpi4py import MPI
from timeit import default_timer as timer
from utils import calculate_interactions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int)
    parser.add_argument('zad', type=int, choices=[1, 2])
    args = parser.parse_args()

    N = args.N
    zad = args.zad

    start_time = timer()

    comm = MPI.COMM_WORLD
    p = comm.Get_size()
    rank = comm.Get_rank()

    assert N % p == 0  # TODO: support number of stars indivisible by number of processes
    n = N // p

    left = (rank - 1) % p
    right = (rank + 1) % p

    # Każdy z procesów zawiera równoliczny zbiór gwiazd N/p, ...
    start = rank * n
    end = (rank + 1) * n
    stars_own = np.load('stars.npy')[start:end, :]  # M, x, y, z

    # ...bufor na gwiazdy przechodnie...
    # Początkowo w buforze przechodnim znajdują się własne gwazdy danego procesu.
    stars_buffer = np.concatenate([stars_own] * 2)  # double buffering
    stars_buffer_A = stars_buffer[:n]
    stars_buffer_B = stars_buffer[n:]

    # ...bufor na akumulator przechodni...
    if zad == 2:
        accumulator_buffer = np.zeros((n * 2, 3))
        accumulator_buffer_A = accumulator_buffer[:n]
        accumulator_buffer_B = accumulator_buffer[n:]

    # ...oraz akumulator swoich interakcji.
    interactions = calculate_interactions(stars_own, stars_own)
    masses_own = stars_own[:, 0, np.newaxis]
    accumulator = (interactions * masses_own).sum(axis=1)  # ax, ay, az

    # Każdy z procesów powtarza p-1 razy następujące czynności:
    # Następnie powtarza floor(p/2) razy następujące czynności:
    iters = p // 2 if zad == 2 else p-1
    for i in range(iters):

        # 1. przesyła przechodnią porcję(N/p) gwiazd do lewego sąsiada w pierścieniu procesów
        comm.Isend([stars_buffer_A, MPI.FLOAT], dest=left)
        if zad == 2:
            # 1. przesyła do lewego sąsiada przechodnią porcję (N/p) gwiazd oraz skojarzony z nimi akumulator przechodni.
            comm.Isend([accumulator_buffer_A, MPI.FLOAT], dest=left)

        # 2. odbiera porcję (N/p) gwiazd od prawego sąsiada do swojego bufora przechodniego
        comm.Recv([stars_buffer_B, MPI.FLOAT], source=right)
        if zad == 2:
            # 2. odbiera porcję (N/p) gwiazd oraz skojarzony z nimi akumulator przechodni od prawego sąsiada do swoich buforów
            comm.Recv([accumulator_buffer_B, MPI.FLOAT], source=right)

        # 3. oblicza interakcje swoich gwiazd i gwiazd otrzymanych
        interactions = calculate_interactions(stars_own, stars_buffer_B)

        # 4. dodaje obliczona interakcje do swojego akumulatora
        masses_buffer = stars_buffer_B[:, 0, np.newaxis]
        accumulator += (interactions * masses_buffer).sum(axis=1)

        if zad == 2 and not (p % 2 == 0 and i == (p // 2) - 1):
            # 5. dodaje obliczone interakcje do bufora przechodniego
            accumulator_buffer_B += (-interactions.transpose(1, 0, 2) * masses_own).sum(axis=1)

        # swap buffers
        stars_buffer_A, stars_buffer_B = stars_buffer_B, stars_buffer_A
        if zad == 2:
            accumulator_buffer_A, accumulator_buffer_B = accumulator_buffer_B, accumulator_buffer_A

    if zad == 2:
        # Po ostatnim kroku wysyła akumulator przechodni do właściciela gwiazd ostatniej przechodniej porcji.
        comm.Isend([accumulator_buffer_A, MPI.FLOAT], dest=(rank - (p+1) // 2) % p)
        comm.Recv([accumulator_buffer_B, MPI.FLOAT], source=(rank + (p+1) // 2) % p)
        accumulator += accumulator_buffer_B

    accelerations = np.empty((p, N // p, 3)) if rank == 0 else None
    comm.Gather(sendbuf=accumulator, recvbuf=accelerations)

    if rank == 0:  # root
        accelerations = accelerations.reshape(N, 3)
        duration = timer() - start_time

        print(accelerations)
        print(accelerations.shape, p, duration)
        np.save(f'interactions_zad{zad}_np{p}.npy', accelerations)
