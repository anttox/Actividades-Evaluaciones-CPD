import asyncio
import pickle
from random import randint
from time import sleep

results = {}
node_loads = {}

# Funcion asincronica para enviar un trabajo
async def submit_job(reader, writer, node_id):
    # Generamos un nuevo ID de trabajo
    job_id = max(list(results.keys()) + [0]) + 1
    # Envia el ID del trabajo al cliente
    writer.write(job_id.to_bytes(4, 'little'))
    await writer.drain()
    sleep_time = randint(1, 4)
    await asyncio.sleep(sleep_time)
    # Guardamos el resultado del trabajo
    results[job_id] = sleep_time
    node_loads[node_id] += 1  # Actualizamos la carga del nodo

# Funcion asincronica para obtener resultados
async def get_results(reader, writer):
    # Recibe el ID del trabajo del cliente
    job_id = int.from_bytes(await reader.read(4), 'little')
    data = pickle.dumps(results.get(job_id, None))
    writer.write(len(data).to_bytes(4, 'little'))
    # Envia el resultado serializado
    writer.write(data)
    # await.drain = Aegura que todos los datos pendientes se hayan enviado a traves del socket antes de continuar
    await writer.drain()
    writer.close()
    await writer.wait_closed()

# Funcion asincronica para obtener la carga del nodo
async def get_load(reader, writer, node_id):
    # Obtenemos la carga del nodo
    load = node_loads.get(node_id, 0)
    # Enviamos la carga del nodo al cliente
    writer.write(load.to_bytes(4, 'little'))
    await writer.drain()
    writer.close()
    await writer.wait_closed()

# Funcion asincronica para manejar las solicitudes de los clientes
async def accept_requests(reader, writer, node_id):
    # Recibe la operacion solicitada por el cliente
    op = await reader.read(1)
    if op[0] == 0:
        await submit_job(reader, writer, node_id)
    elif op[0] == 1:
        await get_results(reader, writer)
    elif op[0] == 2:
        await get_load(reader, writer, node_id)

# Funcion asincronica principal para iniciar el servidor
async def start_server(node_id, host, port):
    server = await asyncio.start_server(lambda r, w: accept_requests(r, w, node_id), host, port)
    async with server:
        await server.serve_forever()

async def main():
    nodes = [
        ('127.0.0.1', 1936, 'node1'),
        ('127.0.0.1', 1937, 'node2'),
        ('127.0.0.1', 1938, 'node3')
    ]
    # Itera sobre cada nodo y crea una tarea para iniciar el servidor
    for host, port, node_id in nodes:
        # Inicia la carga de cada nodo
        node_loads[node_id] = 0
        asyncio.create_task(start_server(node_id, host, port))
    # Bucle infinito para mantener el programa en ejecucion
    while True:
        # Pausamos la ejecucion durante 1 hora (simulamos una espera indefinida)
        await asyncio.sleep(3600)

asyncio.run(main())
