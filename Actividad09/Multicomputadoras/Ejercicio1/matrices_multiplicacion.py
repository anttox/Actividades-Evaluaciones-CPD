from multiprocessing import Pool  # Para manejar el paralelismo utilizamos multiprocessing
import numpy as np  # Para manejar operaciones matriciales de manera eficiente usamos numpy

# Funcion para multiplicar un segmento de la matriz A con la matriz B
def matrix_multiply_segment(args):
    A_segment, B = args
    return np.dot(A_segment, B)  # Multiplicacion de matrices usando numpy

if __name__ == "__main__":
    # Iniciamps las matrices A y B con valores aleatorios
    A = np.random.rand(1000, 1000)
    B = np.random.rand(1000, 1000)

    num_processes = 4  # Numero de procesos a utilizar
    segment_size = A.shape[0] // num_processes  # Tamaño de cada segmento de la matriz A

    # Dividimos la matriz A en segmentos para cada proceso
    segments = [(A[i * segment_size:(i + 1) * segment_size], B) for i in range(num_processes)]

    # Creamos un Pool de procesos
    with Pool(num_processes) as pool:
        # Realizamos la multiplicacion de matrices en paralelo
        results = pool.map(matrix_multiply_segment, segments)

    # Combinamos los resultados de todos los procesos para formar la matriz resultante
    C = np.vstack(results)
    
    # Mostramos la matriz resultante
    print("Matrix multiplication completed.")
    print(C)
