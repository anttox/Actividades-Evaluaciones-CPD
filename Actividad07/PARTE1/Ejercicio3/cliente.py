import dill as pickle
import socket
from time import sleep
import asyncio

def my_funs():
    def mapper(v):
        return v, 1

    def reducer(my_args):
        v, obs = my_args
        return v, sum(obs)
    return mapper, reducer

async def do_request(task_type):
    conn = None
    try:
        conn = socket.create_connection(('127.0.0.1', 1936))
        conn.send(b'\x00')  # Indica que es una solicitud de trabajo
        task_type_encoded = task_type.encode('utf-8')  # Codifica la cadena en bytes
        conn.send(len(task_type_encoded).to_bytes(4, 'little'))  # Envía el tamaño de los bytes
        conn.send(task_type_encoded)  # Envía los bytes codificados
        job_id = int.from_bytes(conn.recv(4), 'little')  # Recibe el job_id del servidor
        print(f'Obteniendo datos desde job_id {job_id}')
    finally:
        if conn:
            conn.close()

    result = None
    while result is None:
        try:
            conn = socket.create_connection(('127.0.0.1', 1936))
            conn.send(b'\x01')
            conn.send(job_id.to_bytes(4, 'little'))
            result_size = int.from_bytes(conn.recv(4), 'little')
            result = pickle.loads(conn.recv(result_size))
        finally:
            if conn:
                conn.close()
        await asyncio.sleep(1)
    print(f'El resultado es {result}')

async def main():
    await asyncio.gather(
        do_request('simple'),
        do_request('moderate'),
        do_request('complex')
    )

if __name__ == '__main__':
    asyncio.run(main())
