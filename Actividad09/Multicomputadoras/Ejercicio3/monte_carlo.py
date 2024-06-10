# Comando para correr el codigo: mpiexec -n 4 python monte_carlo.py
# Importamos el módulo MPI del paquete mpi4py
from mpi4py import MPI
import numpy as np

# Iniciamos el comunicador MPI
comm = MPI.COMM_WORLD
# Obtenemos el rango (identificador unico) del proceso actual
rank = comm.Get_rank()
# Obtenemos el tamaño del comunicador (numero total de procesos)
size = comm.Get_size()

# Se define el numero total de muestras para utilizar en la simulacion de Monte Carlo
num_samples = 1000000
# Calculamos el numero de muestras que cada proceso va a manejar
local_samples = num_samples // size

# Establecemos la semilla para la generacion de numeros aleatorios, basado en el rango del proceso
np.random.seed(rank)
# Iniciamos el contador local de puntos dentro del circulo
local_count = 0

# Generamos puntos aleatorios y contamos cuantos caen dentro del circulo unitario
for _ in range(local_samples):
    x, y = np.random.rand(2)
    if x**2 + y**2 <= 1.0:
        local_count += 1

# Reducimos los contadores locales a un contador total en el proceso raiz (0)
total_count = comm.reduce(local_count, op=MPI.SUM, root=0)

# El proceso raiz calcula y muestra la estimacion de pi
if rank == 0:
    pi_estimate = (4.0 * total_count) / num_samples
    print(f"Estimated Pi: {pi_estimate}")

