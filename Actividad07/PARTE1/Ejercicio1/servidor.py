# Servidor
# asyncio = concurrencia
import asyncio
#import marshal
import pickle
from random import randint
# Codigo de Tiago Rodriguez

results = {}

# Funcion asincrona que maneja la creacion de nuevos trabajos
async def submit_job(reader, writer):
    job_id = max(list(results.keys()) + [0]) + 1
    #codificacion
    writer.write(job_id.to_bytes(4, 'little'))
    #cerrar la conexion de escritura
    writer.close()
    sleep_time = randint(1, 4)
    await asyncio.sleep(sleep_time)
    results[job_id] = sleep_time


# Funcion que maneja la obtencion de resultados de los nuevos trabajos
async def get_results(reader, writer):
    job_id = int.from_bytes(await reader.read(4), 'little')
    data = pickle.dumps(results.get(job_id, None))
    # Envia el tamaño de los datos serializados al cliente en 4 bytes
    writer.write(len(data).to_bytes(4, 'little'))
    # Envia los datos serializados al cliente
    writer.write(data)

async def sum_numbers(reader, writer):
    job_id = max(list(results.keys()) + [0]) + 1
    writer.write(job_id.to_bytes(4, 'little'))
    # Asegura que todos los datos se han enviado antes de cerrar la conexión
    await writer.drain()
    length = int.from_bytes(await reader.read(4), 'little')
    data = pickle.loads(await reader.read(length))
    result = sum(data)
    results[job_id] = result


# Funcion asincrona que maneja las solicitudes entrantes
async def accept_requests(reader, writer):
    # Lee un byte del cliente que indica que funcion usar 0 = submit_jod, 1 = get_results
    op = await reader.read(1)
    if op[0] == 0:
        await submit_job(reader, writer)
    elif op[0] == 1:
        await get_results(reader, writer)
    # Leer el byte del cliente para usar la funcion send_list
    elif op[0] == 2:
        await sum_numbers(reader, writer)
        
async def main():
    server = await asyncio.start_server(accept_requests, '127.0.0.1', 1936)
    # Asegura que el servidor se cierre correctamente al finalizar
    async with server:
        # Hace que el servidor sirva indefinidamente
        await server.serve_forever()

asyncio.run(main())