import time
import numpy as np
import random

# Funcion que implementa el prefetching de datos
def prefetch_data(array, indices, prefetch_distance):
    for i in range(len(indices) - prefetch_distance):
        _ = array[indices[i + prefetch_distance]]  # Prefetch
        array[indices[i]] += 1  # Access

# Definimos el tamaño del array
size = 10**6
array = np.zeros(size)

# Creamos una lista de indices aleatorios
indices = list(range(size))
random.shuffle(indices)

# Distancia de prefetching
prefetch_distance = 10

# Medimos el tiempo de inicio
start_time = time.perf_counter()

# Ejecutamos la funcion de prefetching
prefetch_data(array, indices, prefetch_distance)

# Medimos el tiempo de finalizacion
end_time = time.perf_counter()

# Calculamos el tiempo total
total_time = end_time - start_time

# Imprimimos el tiempo total de ejecucion con prefetching
print(f"Tiempo con prefetching: {total_time:.6f} segundos")
