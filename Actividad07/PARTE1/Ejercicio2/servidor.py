import asyncio
import pickle
from random import randint

results = {}

async def submit_job(reader, writer):
    job_id = max(list(results.keys()) + [0]) + 1
    writer.write(job_id.to_bytes(4, 'little'))
    await writer.drain()  # Asegurarse de que los datos se envíen
    sleep_time = randint(1, 4)
    await asyncio.sleep(sleep_time)
    results[job_id] = sleep_time

async def get_results(reader, writer):
    job_id = int.from_bytes(await reader.read(4), 'little')
    data = pickle.dumps(results.get(job_id, None))
    writer.write(len(data).to_bytes(4, 'little'))
    writer.write(data)
    await writer.drain()  # Asegurarse de que los datos se envíen

async def accept_requests(reader, writer):
    try:
        op = await reader.read(1)
        if op[0] == 0:
            await submit_job(reader, writer)
        elif op[0] == 1:
            await get_results(reader, writer)
    # except captura cualquier excepción que se produzca durante la ejecucion del codigo en el bloque try
    # La variable e contiene la información sobre la excepción, lo que permite registrar o imprimir detalles del error para facilitar la depuracion
    except Exception as e:
        print(f'Error handling request: {e}')
    finally:
        writer.close()

async def main():
    server = await asyncio.start_server(accept_requests, '127.0.0.1', 1936)
    async with server:
        await server.serve_forever()

asyncio.run(main())

