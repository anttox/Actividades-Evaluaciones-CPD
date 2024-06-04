import multiprocessing
import numpy as np
import time
import psutil # Para establecer la afinidad de CPU del proceso

# Funcion para asignar un nucleo especifico a un proceso y realiza operaciones en datos compartidos
# core_id (int): ID del nucleo donde se asignara el proceso.
# shared_data: arreglo compartido donde se realizaran las operaciones
def core_affinity_worker(core_id, shared_data, start, end):
    # Establecemos la afinidad del nucleo utilizando psutil
    p = psutil.Process()
    p.cpu_affinity([core_id])
    
    # Realizamos una operacion en el segmento de datos asignado
    for i in range(start, end):
        shared_data[i] += 1

# Tamaño del arreglo de datos compartidos
size = 10**6
shared_data = np.zeros(size)
# Numero de procesos a utilizar
num_processes = 4
processes = []

# Calculamos el tamaño de cada segmento de datos para cada proceso
chunk_size = size // num_processes
for i in range(num_processes):
    start = i * chunk_size
    end = (i + 1) * chunk_size
    # Creamos un nuevo proceso asignado a un nucleo especifico
    process = multiprocessing.Process(target=core_affinity_worker, args=(i, shared_data, start, end))
    processes.append(process)

# Meimos el tiempo de inicio
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

print(f"Tiempo total con afinidad de núcleo y {num_processes} procesos: {total_time:.6f} segundos")
