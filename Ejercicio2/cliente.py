# Pickle no serializa pero dill si lo hace
import dill as pickle  # Usamos dill en lugar de pickle
import socket
from time import sleep

def mapper(v):
    return v, 1

def reducer(my_args):
    v, obs = my_args
    return v, sum(obs)
# Serializacion con dumps
def do_request(data):
    conn = None
    try:
        conn = socket.create_connection(('127.0.0.1', 1940))  # Cambié el puerto a 1940
        conn.send(b'\x00')
        mapper_code = pickle.dumps(mapper)
        reducer_code = pickle.dumps(reducer)
        conn.send(len(mapper_code).to_bytes(4, 'little', signed=False))
        conn.send(mapper_code)
        conn.send(len(reducer_code).to_bytes(4, 'little'))
        conn.send(reducer_code)
        data_code = pickle.dumps(data)
        conn.send(len(data_code).to_bytes(4, 'little'))
        conn.send(data_code)
        job_id = int.from_bytes(conn.recv(4), 'little')
        print(f'Obteniendo datos desde job_id {job_id}')
    finally:
        if conn:
            conn.close()

    result = None
    while result is None:
        try:
            conn = socket.create_connection(('127.0.0.1', 1940))  # Cambié el puerto a 1940
            conn.send(b'\x01')
            conn.send(job_id.to_bytes(4, 'little'))
            result_size = int.from_bytes(conn.recv(4), 'little')
            result = pickle.loads(conn.recv(result_size))
        finally:
            if conn:
                conn.close()
        sleep(1)
    print(f'El resultado es {result}')

if __name__ == '__main__':
    do_request('Python rocks. Python es lo maximo'.split(' '))

