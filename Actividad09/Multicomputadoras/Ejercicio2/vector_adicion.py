from mpi4py import MPI  # Para manejar la comunicacion entre procesos utilizanmos MPI
import numpy as np  # Para manejar operaciones con vectores 

# Iniciamos el comunicador MPI
comm = MPI.COMM_WORLD

# Obtenemos el rango (identificador unico) del proceso actual
rank = comm.Get_rank()

# Obtenemos el tamaño del comunicador (numero total de procesos)
size = comm.Get_size()

# Definimos el tamaño total del vector
vector_size = 1000000

# Calculamos el tamaño del segmento de cada proceso
local_size = vector_size // size

# El proceso de rango 0 inicia los vectores A y B
if rank == 0:
    A = np.random.rand(vector_size)
    B = np.random.rand(vector_size)
else:
    A = None
    B = None

# Iniciamos los segmentos locales de los vectores
local_A = np.empty(local_size, dtype='d')
local_B = np.empty(local_size, dtype='d')

# Distribuimos los segmentos de los vectores A y B a todos los procesos
comm.Scatter(A, local_A, root=0)
comm.Scatter(B, local_B, root=0)

# Realizamos la suma de los segmentos locales
local_C = local_A + local_B

# El proceso de rango 0 inicia el vector resultante C
if rank == 0:
    C = np.empty(vector_size, dtype='d')
else:
    C = None

# Recogemos y combinamos los resultados de todos los procesos en el proceso raiz
comm.Gather(local_C, C, root=0)

# El proceso de rango 0 imprime un mensaje cuando la suma de vectores se ha completado
if rank == 0:
    print("Vector addition completed. Result vector C:")
    print(C)
