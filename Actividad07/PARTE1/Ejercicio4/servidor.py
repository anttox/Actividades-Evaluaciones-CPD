import asyncio
import pickle
from concurrent.futures import ProcessPoolExecutor

results = {}
executor = ProcessPoolExecutor()

def cpu_intensive_task(task_type):
    if task_type == "simple":
        return sum(i * i for i in range(1000000))  # Tarea simple
    elif task_type == "moderate":
        return sum(i * i for i in range(10000000))  # Tarea moderada
    elif task_type == "complex":
        return sum(i * i for i in range(100000000))  # Tarea compleja

async def submit_job(reader, writer):
    job_id = max(list(results.keys()) + [0]) + 1
    writer.write(job_id.to_bytes(4, 'little'))
    await writer.drain()
    
    task_type_size = int.from_bytes(await reader.read(4), 'little')
    task_type = (await reader.read(task_type_size)).decode('utf-8')

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, cpu_intensive_task, task_type)
    
    results[job_id] = result
    writer.close()

async def get_results(reader, writer):
    job_id = int.from_bytes(await reader.read(4), 'little')
    data = pickle.dumps(results.get(job_id, None))
    writer.write(len(data).to_bytes(4, 'little'))
    writer.write(data)
    await writer.drain()
    writer.close()

async def accept_requests(reader, writer):
    try:
        op = await reader.read(1)
        if op[0] == 0:
            await submit_job(reader, writer)
        elif op[0] == 1:
            await get_results(reader, writer)
    except Exception as e:
        print(f'Error handling request: {e}')
    finally:
        writer.close()

async def main():
    server = await asyncio.start_server(accept_requests, '127.0.0.1', 1936)
    async with server:
        await server.serve_forever()

asyncio.run(main())
