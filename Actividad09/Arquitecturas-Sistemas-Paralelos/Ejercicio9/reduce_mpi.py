# Comando para correr el codigo: mpiexec -n 4 python reduce_mpi.py
from mpi4py import MPI

# Iniciamos el comunicador MPI
comm = MPI.COMM_WORLD

# Obtenemos el rango del proceso (identificador unico del proceso dentro del comunicador)
rank = comm.Get_rank()

# Cada proceso tiene un numero diferente
number = rank + 1  # El número de cada proceso es su rango + 1

# Se realiza una reduccion para sumar los numeros de todos los procesos
total_sum = comm.reduce(number, op=MPI.SUM, root=0)

# El proceso de rango 0 imprime la suma total
if rank == 0:
    print(f"Total sum: {total_sum}")

