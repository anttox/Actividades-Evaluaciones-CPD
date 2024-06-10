# Comando para correr el codigo: mpiexec -n 4 python correct_word_count.py
from mpi4py import MPI  # Para manejar la comunicacion entre procesos utilizamos MPI

# Iniciamos el comunicador MPI
comm = MPI.COMM_WORLD

# Obtenemos el rango (identificador unico) del proceso actual
rank = comm.Get_rank()

# Obtenemos el tamaño del comunicador (numero total de procesos)
size = comm.Get_size()

# El proceso de rango 0 lee el archivo de texto grande y distribuye las lineas
if rank == 0:
    with open('cenicienta.txt', 'r') as file:
        lines = file.readlines()
    # Dividimos las lineas en segmentos para cada proceso
    segments = [lines[i::size] for i in range(size)] # Cambio del anterior codigo
else:
    segments = None

# Se distribuye los segmentos de lineas a todos los procesos
lines = comm.scatter(segments, root=0)

# Cada proceso cuenta las palabras en sus lineas locales
local_word_count = sum(len(line.split()) for line in lines)

# Reducimos los contadores locales a un contador total en el proceso raiz (0)
total_word_count = comm.reduce(local_word_count, op=MPI.SUM, root=0)

# El proceso de rango 0 imprime el conteo total de palabras
if rank == 0:
    print(f"Total word count: {total_word_count}")
