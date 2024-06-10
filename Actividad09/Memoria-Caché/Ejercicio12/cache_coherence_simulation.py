import multiprocessing
import numpy as np
import time

# Funcion que simula la coherencia de cache en un entorno multinucleo
def coherence_worker(shared_array, start, end):
    for i in range(start, end):
        shared_array[i] += 1  # Se modifica el valor del array compartido

# Tamaño del array compartido
size = 10**6
shared_array = multiprocessing.Array('d', size)  # Creamos un array compartido de tipo doble

# Numero de procesos a utilizar
num_processes = 4
chunk_size = size // num_processes  # Tamaño del segmento de datos para cada proceso
processes = []

# Creamos y asignamos procesos a diferentes segmentos del array
for i in range(num_processes):
    start = i * chunk_size
    end = (i + 1) * chunk_size
    process = multiprocessing.Process(target=coherence_worker, args=(shared_array, start, end))
    processes.append(process)

# Medimos el tiempo de inicio
start_time = time.perf_counter()

# Iniciamos todos los procesos
for process in processes:
    process.start()

# Esperamos a que todos los procesos terminen
for process in processes:
    process.join()

# Medimos el tiempo de fin
end_time = time.perf_counter()
total_time = end_time - start_time

# Imprimimos el tiempo total de ejecucion
print(f"Tiempo total con coherencia de caché y {num_processes} procesos: {total_time:.6f} segundos")
