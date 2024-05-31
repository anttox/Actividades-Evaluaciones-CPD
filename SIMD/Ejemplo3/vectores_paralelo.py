import numpy as np

# Definimos la funcion para sumar dos vectores usando NumPy
def vector_addition(a, b):
    # Utilizamos np.add para sumar los dos arreglos
    return np.add(a, b)

# Creamos dos arreglos de NumPy con valores enteros
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# Llamamos a la funcion vector_addition para sumar los arreglos a y b
result = vector_addition(a, b)

# Imprimimos el resultado de la suma de los arreglos
print(result)

