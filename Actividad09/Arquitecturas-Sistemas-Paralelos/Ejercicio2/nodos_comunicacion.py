# Comando para correr el programa en la terminal: mpiexec -n 4 python nodos_comunicacion.py
# Este comando ejecuta el programa con 4 procesos. Cada proceso ejecuta el codigo y se comunica segun lo que se defina
from mpi4py import MPI

# Inicializamos el comunicador
comm = MPI.COMM_WORLD

# Obtenemos el rango del proceso (identificador unico del proceso)
rank = comm.Get_rank()

# Obtenemos el tamaño del comunicador (numero total de procesos)
size = comm.Get_size()

# Si el proceso es el de rango 0, envía un mensaje a todos los demas procesos
if rank == 0:
    data = "Hola desde el rango 0"
    for i in range(1, size):
        comm.send(data, dest=i)  # Envia el mensaje al proceso con el rango 'i'
else:
    # Si el proceso no es el de rango 0, recibe el mensaje del proceso de rango 0
    data = comm.recv(source=0)
    print(f"El rango {rank} recibió datos: {data}")  # Imprime el mensaje recibido

