from mpi4py import MPI
from threading import Thread

# Funcion que ejecuta una tarea en un hilo
def tarea_hilo(rango):
    print(f"Hilo en rango {rango} está ejecutándose")

# Iniciamos el comunicador
comm = MPI.COMM_WORLD

# Obtenemos el rango del proceso (identificador unico del proceso)
rango = comm.Get_rank()

# Obtenemos el tamaño del comunicador (numero total de procesos)
tamano = comm.Get_size()

# Lista para almacenar los hilos
hilos = []

# Creamos y lanzamos 4 hilos en cada proceso
for _ in range(4):
    t = Thread(target=tarea_hilo, args=(rango,))
    hilos.append(t)
    t.start()

# Esperamos que todos los hilos terminen
for t in hilos:
    t.join()

# Sincronizar todos los procesos
comm.Barrier()

# Mensaje final en el proceso de rango 0
if rango == 0:
    print("Todos los hilos han terminado en todos los nodos.")
