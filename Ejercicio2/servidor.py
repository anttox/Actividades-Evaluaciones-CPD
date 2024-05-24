# Usamos asyncio para manejar la comunicacion asincronica
import asyncio
import dill as pickle  # Usamos dill en lugar de pickle
from collections import defaultdict
# Usamos ThreadPoolExecutor par ejecutar tareas en paralelo
from concurrent.futures import ThreadPoolExecutor
from random import randint

results = {}

# Deserializacion con loads
def execute_map_reduce(job_id, mapper_code, reducer_code, data):
    mapper = pickle.loads(mapper_code)
    reducer = pickle.loads(reducer_code)
    
    with ThreadPoolExecutor() as executor:
        map_results = list(executor.map(mapper, data))
    
    distributor = defaultdict(list)
    for key, value in map_results:
        distributor[key].append(value)
    
    reduce_results = []
    with ThreadPoolExecutor() as executor:
        reduce_results = list(executor.map(reducer, distributor.items()))
    
    results[job_id] = reduce_results

async def submit_job(reader, writer):
    job_id = max(list(results.keys()) + [0]) + 1
    mapper_size = int.from_bytes(await reader.read(4), 'little')
    mapper_code = await reader.read(mapper_size)
    reducer_size = int.from_bytes(await reader.read(4), 'little')
    reducer_code = await reader.read(reducer_size)
    data_size = int.from_bytes(await reader.read(4), 'little')
    data = await reader.read(data_size)
    data = pickle.loads(data)
    
    # Ejecuta execute_map_reduce en un hilo separado usando run_in_executor
    # Bloqueo = loop
    # Esto permite que el servidor siga manejando otras conexiones y solicitudes mientras las tareas de map y reduce se ejecutan en segundo plano
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, execute_map_reduce, job_id, mapper_code, reducer_code, data)
    
    writer.write(job_id.to_bytes(4, 'little'))
    await writer.drain()
    writer.close()

# get_results = Funcion asincronica que maneja la solicitud de resultados
async def get_results(reader, writer):
    job_id = int.from_bytes(await reader.read(4), 'little')
    result = results.get(job_id, None)
    data = pickle.dumps(result)
    writer.write(len(data).to_bytes(4, 'little'))
    writer.write(data)
    await writer.drain()
    writer.close()

async def accept_requests(reader, writer):
    op = await reader.read(1)
    if op[0] == 0:
        await submit_job(reader, writer)
    elif op[0] == 1:
        await get_results(reader, writer)

async def main():
    server = await asyncio.start_server(accept_requests, '127.0.0.1', 1940)  # Cambié el puerto a 1940
    async with server:
        await server.serve_forever()

asyncio.run(main())
