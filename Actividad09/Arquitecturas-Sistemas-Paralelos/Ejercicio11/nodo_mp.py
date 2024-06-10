from mpi4py import MPI # Para manejar la comunicación entre procesos utilizando MPI
from threading import Thread # Para manejar el paralelismo de tareas dentro de cada nodo usando hilos 

# Funcion para calcular el cuadrado y almacenarlo en una lista.
# Indice donde almacenar el resultado de result_list
def compute_square(value, result_list, index):
    result_list[index] = value ** 2

# Iniciamos el comunicador MPI
comm = MPI.COMM_WORLD
# Obtenemos el rango (identificador unico) del proceso actual
rank = comm.Get_rank()
# Obtenemos el tamaño del comunicador (numero total de procesos)
size = comm.Get_size()

data = None
if rank == 0:
    # El proceso raiz (rank 0) prepara los datos para distribuir
    data = list(range(1, 101))  # Datos de ejemplo del 1 al 100
    segment_size = len(data) // size
    # Dividimos los datos en segmentos para cada proceso
    data_segments = [data[i * segment_size:(i + 1) * segment_size] for i in range(size)]
else:
    data_segments = None

# Distribuimos los segmentos de datos a todos los procesos
data_segment = comm.scatter(data_segments, root=0)

# Iniciamos la lista de resultados para el segmento de datos recibido
results = [0] * len(data_segment)
threads = []

# Paralelismo de tareas usando hilos para calcular el cuadrado de cada valor en el segmento
for i in range(len(data_segment)):
    t = Thread(target=compute_square, args=(data_segment[i], results, i))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

# Recogemos y combinamos los resultados de todos los procesos en el proceso raiz
gathered_results = comm.gather(results, root=0)

if rank == 0:
    # El proceso raiz combina los resultados de todos los procesos
    final_results = [item for sublist in gathered_results for item in sublist]
    print(f"Final results: {final_results}")

