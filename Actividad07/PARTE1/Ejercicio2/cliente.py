import dill as pickle
import socket
from time import sleep

def my_funs():
    def mapper(v):
        return v, 1

    def reducer(my_args):
        v, obs = my_args
        return v, sum(obs)
    return mapper, reducer

def do_request(my_funs, data):
    conn = None
    try:
        conn = socket.create_connection(('127.0.0.1', 1936))
        conn.send(b'\x00')
        my_code = pickle.dumps(my_funs.__code__)
        conn.send(len(my_code).to_bytes(4, 'little', signed=False))
        conn.send(my_code)
        my_data = pickle.dumps(data)
        conn.send(len(my_data).to_bytes(4, 'little'))
        conn.send(my_data)
        job_id = int.from_bytes(conn.recv(4), 'little')
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
        sleep(1)
    print(f'El resultado es {result}')

if __name__ == '__main__':
    do_request(my_funs, 'Python rocks. Python es lo maximo'.split(' '))

