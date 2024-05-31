import numpy as np
from joblib import Parallel, delayed # Se utiliza joblid Parallel y delayed para multiplicar sub-matrices en paralelo

# Funcion para multiplicar dos sub-matrices
def multiplicar_sub_matrices(A, B):
    return np.dot(A, B)

# Funcion principal para la multiplicacion paralela de matrices
def multiplicacion_matricial_paralela():
    # Creamos dos matrices grandes aleatorias gracias a np.random.rand
    A = np.random.rand(1000, 1000)
    B = np.random.rand(1000, 1000)
    
    # Dividimos la matriz A en 4 sub-matrices a lo largo de las filas -> axis = 0
    A_subs = np.array_split(A, 4, axis=0)
    
    # Dividimos la matriz B en 4 sub-matrices a lo largo de las columnas -> axis = 1
    B_subs = np.array_split(B, 4, axis=1)

    # Se hace la multiplicacion de las sub-matrices en paralelo
    resultados = Parallel(n_jobs=4)(delayed(multiplicar_sub_matrices)(A_sub, B_sub) for A_sub in A_subs for B_sub in B_subs)
    
    # Se crea una matriz de ceros para almacenar el resultado
    C = np.zeros((1000, 1000))
    
    # Reunimos los resultados y se forma la matriz resultante
    for i in range(4):
        for j in range(4):
            C[i*250:(i+1)*250, j*250:(j+1)*250] = resultados[i*4 + j]

    return C

# Se llama a la funcion principal para obtener la matriz resultante C
C = multiplicacion_matricial_paralela()

# Pequeño ejemplo de los primeros 5 x 5 elementos de la matriz resultante C
print(C[:5, :5])
