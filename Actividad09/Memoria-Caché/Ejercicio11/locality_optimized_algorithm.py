import numpy as np
import time

# Algoritmo optimizado para mejorar la localidad temporal y espacial
def locality_optimized_algorithm(matrix):
    n = len(matrix)  # Obtenemos el tamaño de la matriz
    result = np.zeros((n, n))  # Iniciamos la matriz de resultado con ceros
    
    # Recorremos la matriz para mejorar la localidad temporal y espacial
    for i in range(n):
        for j in range(n):
            result[i][j] = matrix[i][j] + matrix[j][i]  # Accedemos a los elementos en orden para mejorar la localidad
    
    return result  # Retornamos la matriz de resultado

# Definimos el tamaño de la matriz
n = 1000
matrix = np.random.rand(n, n)  # Creamos una matriz aleatoria de tamaño nxn

# Medimos el tiempo de ejecucion del algoritmo optimizado
start_time = time.perf_counter()
result = locality_optimized_algorithm(matrix)
end_time = time.perf_counter()

# Imprimimos el tiempo de ejecucion
print(f"Tiempo de ejecución: {end_time - start_time:.6f} segundos")
