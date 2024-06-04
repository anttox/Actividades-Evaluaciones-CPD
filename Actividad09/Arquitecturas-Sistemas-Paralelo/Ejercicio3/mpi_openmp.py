from mpi4py import MPI
from threading import Thread

# Función que ejecuta una tarea en un hilo
def tarea_hilo(rango):
    print(f"Hilo en rango {rango} está ejecutándose")

# Inicializa el comunicador
comm = MPI.COMM_WORLD

# Obtiene el rango del proceso (identificador único del proceso)
rango = comm.Get_rank()

# Obtiene el tamaño del comunicador (número total de procesos)
tamano = comm.Get_size()

# Lista para almacenar los hilos
hilos = []

# Crea y lanza 4 hilos en cada proceso
for _ in range(4):
    t = Thread(target=tarea_hilo, args=(rango,))
    hilos.append(t)
    t.start()

# Espera a que todos los hilos terminen
for t in hilos:
    t.join()

# Mensaje final en el proceso de rango 0
if rango == 0:
    print("Todos los hilos han terminado en todos los nodos.")

