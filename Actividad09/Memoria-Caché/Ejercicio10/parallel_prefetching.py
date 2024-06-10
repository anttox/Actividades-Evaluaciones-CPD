from concurrent.futures import ThreadPoolExecutor
import numpy as np
import random

# Funcion que realiza el prefetching de datos en un entorno paralelo
def prefetching_worker(array, indices, prefetch_distance):
    for i in range(len(indices) - prefetch_distance):
        _ = array[indices[i + prefetch_distance]]  # Prefetch
        array[indices[i]] += 1  # Access

# Tamaño del array
size = 10**6
array = np.zeros(size)

# Creamos una lista de indices aleatorios
indices = list(range(size))
random.shuffle(indices)

# Distancia de el prefetching
prefetch_distance = 10

# Ejecutamos el prefetching en paralelo utilizando ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as executor:
    # Dividimos el trabajo en 4 partes y asignamos a los trabajadores
    futures = [executor.submit(prefetching_worker, array, indices[i::4], prefetch_distance) for i in range(4)]
    
    # Esperamos que todos los trabajadores completen su trabajo
    for future in futures:
        future.result()

print(f"Prefetching completado")

# Mostramos una parte del array para verificar el resultado de las operaciones
print(f"Parte del array despues del prefetching: {array[:20]}")
