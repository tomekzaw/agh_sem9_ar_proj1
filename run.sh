mpiexec -n 8 --oversubscribe python3 main.py 840 1 &&
mpiexec -n 7 --oversubscribe python3 main.py 840 1 &&
mpiexec -n 6 --oversubscribe python3 main.py 840 1 &&
mpiexec -n 5 --oversubscribe python3 main.py 840 1 &&
mpiexec -n 4 python3 main.py 840 1 &&
mpiexec -n 3 python3 main.py 840 1 &&
mpiexec -n 2 python3 main.py 840 1 &&
mpiexec -n 1 python3 main.py 840 1 &&
mpiexec -n 8 --oversubscribe python3 main.py 840 2 &&
mpiexec -n 7 --oversubscribe python3 main.py 840 2 &&
mpiexec -n 6 --oversubscribe python3 main.py 840 2 &&
mpiexec -n 5 --oversubscribe python3 main.py 840 2 &&
mpiexec -n 4 python3 main.py 840 2 &&
mpiexec -n 3 python3 main.py 840 2 &&
mpiexec -n 2 python3 main.py 840 2 &&
mpiexec -n 1 python3 main.py 840 2
