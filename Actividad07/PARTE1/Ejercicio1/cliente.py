import dill as pickle
import socket
from time import sleep

def send_list(data):
    conn = None
    try:
        conn = socket.create_connection(('127.0.0.1', 1936))
        conn.send(b'\x02')  # Nuevo código de operación para enviar una lista
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
    # Enviar lista de números para sumar
    send_list([1, 2, 3, 4, 5])
