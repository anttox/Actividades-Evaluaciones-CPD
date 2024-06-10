import threading
import time
import numpy as np

# Funcion que cada hilo va a ejecutar, modificando una porcion de shared_data
def worker(shared_data, start, end):
    for i in range(start, end):
        shared_data[i] += 1

# Definimos el tamaño del array compartido
size = 10**6
shared_data = np.zeros(size)

# Definimos el numero de hilos
num_threads = 4
threads = []

# Calculamos el tamaño del segmento de datos que cada hilo procesara
chunk_size = size // num_threads

# Creamos y lanzamos los hilos
for i in range(num_threads):
    start = i * chunk_size
    end = (i + 1) * chunk_size
    thread = threading.Thread(target=worker, args=(shared_data, start, end))
    threads.append(thread)

# Medimos el tiempo de inicio
start_time = time.perf_counter()

# Iniciamos todos los hilos
for thread in threads:
    thread.start()

# Esperamos a que todos los hilos terminen
for thread in threads:
    thread.join()

# Medimos el tiempo de finalizacion
end_time = time.perf_counter()

# Calculamos el tiempo total
total_time = end_time - start_time

# Imprimimos el tiempo total de ejecucion
print(f"Tiempo total con {num_threads} hilos: {total_time:.6f} segundos")
