from mpi4py import MPI
from threading import Thread

# Funcion que procesa un segmento de datos
def process_data(data_segment):
    result = [x ** 2 for x in data_segment]  # Calculamos el cuadrado de cada elemento en el segmento
    print(f"Processed segment: {result}")

# Iniciamos el comunicador MPI
comm = MPI.COMM_WORLD

# Metodos esenciales para coordinar la distribucion de trabajo entre los procesos (rank y size)
# Obtenemos el rango del proceso (identificador unico del proceso dentro del comunicador)
rank = comm.Get_rank()

# Obtenemos el tamaño del comunicador (numero total de procesos)
size = comm.Get_size()

data = None

# El proceso de rango 0 inicia los datos y los divide en segmentos
if rank == 0:
    data = list(range(100))  # Lista de 100 elementos
    segment_size = len(data) // size  # Tamaño de cada segmento
    data_segments = [data[i * segment_size:(i + 1) * segment_size] for i in range(size)]
else:
    data_segments = None

# Se distribuye los segmentos de datos a cada proceso
data_segment = comm.scatter(data_segments, root=0)

# Creamos y lanzamos 4 hilos en cada proceso para procesar los datos en paralelo
threads = []
for i in range(4):
    t = Thread(target=process_data, args=(data_segment[i::4],))
    threads.append(t)
    t.start()

# Esperamos que todos los hilos terminen
for t in threads:
    t.join()

# Sincroniza todos los procesos MPI
comm.Barrier()

# El proceso de rango 0 imprime un mensaje cuando todos los datos han sido procesados
if rank == 0:
    print("All data processed.")
