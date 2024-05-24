import asyncio
import pickle
from random import randint
import time

# Diccionario para almacenar resultados de los trabajos
results = {}
# Diccionario para almacenar el estado de los nodos (ultimo "heartbeat")
node_status = {}
# Diccionario para almacenar las tareas asignadas a cada nodo
tasks = {}

# Funcion para monitorear los "heartbeats" de los nodos
def monitor_heartbeat(node_id):
    while True:
        time.sleep(5)
        if node_id in node_status and time.time() - node_status[node_id] > 10:
            print(f'Node {node_id} failed. Reassigning tasks...')
            reassign_tasks(node_id)

# Funcion para reasignar tareas en caso de fallo de un nodo
def reassign_tasks(node_id):
    if node_id in tasks:
        for job_id, task in tasks[node_id]:
            new_node_id = get_available_node()
            tasks[new_node_id].append((job_id, task))
            # Aquí se replican los datos si es necesario
            print(f'Task {job_id} reassigned to node {new_node_id}')

# Funcion para obtener un nodo disponible aleatoriamente (parte simulada)
def get_available_node():
    return randint(1, 5)  # Asumiendo que hay 5 nodos disponibles

# Funcion para manejar la sumision (envio) de trabajos
async def submit_job(reader, writer):
    job_id = max(list(results.keys()) + [0]) + 1
    writer.write(job_id.to_bytes(4, 'little'))
    writer.close()
    sleep_time = randint(1, 4)
    await asyncio.sleep(sleep_time)
    results[job_id] = sleep_time

# Funcion para obtener los resultados de un trabajo
async def get_results(reader, writer):
    job_id = int.from_bytes(await reader.read(4), 'little')
    data = pickle.dumps(results.get(job_id, None))
    writer.write(len(data).to_bytes(4, 'little'))
    writer.write(data)

# Funcion para aceptar solicitudes del cliente
async def accept_requests(reader, writer):
    op = await reader.read(1)
    if op[0] == 0:
        await submit_job(reader, writer)
    elif op[0] == 1:
        await get_results(reader, writer)

# Funcion para manejar los "heartbeats" de los nodos
async def heartbeat(reader, writer):
    node_id = int.from_bytes(await reader.read(4), 'little')
    node_status[node_id] = time.time()
    print(f'Received heartbeat from node {node_id}')
    writer.write(b'OK')

# Funcion principal para iniciar el servidor
async def main():
    server = await asyncio.start_server(accept_requests, '127.0.0.1', 1936)
    async with server:
        await server.serve_forever()

asyncio.run(main())

