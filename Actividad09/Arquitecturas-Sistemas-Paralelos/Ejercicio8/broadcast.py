# Comando para correr el codigo: mpiexec -n 4 python broadcast.py
from mpi4py import MPI

# Iniciamos el comunicador MPI
comm = MPI.COMM_WORLD

# Obtenemos el rango del proceso (identificador unico del proceso dentro del comunicador)
rank = comm.Get_rank()

# El proceso con rango 0 define los datos a enviar
if rank == 0:
    data = {'key1': 'value1', 'key2': 'value2'}
else:
    # Los demas procesos inician data como None
    data = None

# Se realiza un broadcast de los datos desde el proceso de rango 0 a todos los demás procesos
data = comm.bcast(data, root=0)

# Cada proceso imprime los datos recibidos
print(f"Rank {rank} received data: {data}")

