import time
import numpy as np
import random

# Funcion para medir el tiempo de acceso a los datos en un array basado en un conjunto de indices
def measure_access_time(array, indices):
    # Registramos el tiempo de inicio
    start_time = time.perf_counter()
    
    # Accedemos a los elementos del array segun los indices proporcionados
    for i in indices:
        array[i] += 1
    
    # Registramos el tiempo de finalizacion
    end_time = time.perf_counter()
    
    # Calculamos el tiempo de acceso total
    access_time = end_time - start_time
    return access_time

# Definimos el tamaño del array
size = 10**6
array = np.zeros(size)

# Creamos indices secuenciales y aleatorios
sequential_indices = list(range(size))
random_indices = sequential_indices.copy()
random.shuffle(random_indices)

# Medimos el tiempo de acceso secuencial
sequential_time = measure_access_time(array, sequential_indices)

# Medimos el tiempo de acceso aleatorio
random_time = measure_access_time(array, random_indices)

# Imprimimos los resultados de los tiempos de acceso
print(f"Tiempo de acceso secuencial: {sequential_time:.6f} segundos")
print(f"Tiempo de acceso aleatorio: {random_time:.6f} segundos")
