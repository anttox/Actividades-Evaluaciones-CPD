import dill as pickle
import socket
from time import sleep

def my_funs():
    # Funcion de mapeo
    def mapper(v):
        return v, 1

    # Funcion de reduccion
    def reducer(my_args):
        v, obs = my_args
        return v, sum(obs)
    return mapper, reducer

def do_request(my_funs, data):
    conn = None
    try:
        # Conecta al servidor
        conn = socket.create_connection(('127.0.0.1', 1936))
        conn.send(b'\x00')  # Enviar codigo de operacion para "submit job"
        
        # Serializa y envia las funciones de mapeo y reducción
        my_code = pickle.dumps(my_funs.__code__)
        conn.send(len(my_code).to_bytes(4, 'little', signed=False))
        conn.send(my_code)
        
        # Serializa y envia los datos
        my_data = pickle.dumps(data)
        conn.send(len(my_data).to_bytes(4, 'little'))
        conn.send(my_data)
        
        # Recibe el job_id asignado por el servidor
        job_id = int.from_bytes(conn.recv(4), 'little')
        print(f'Obteniendo datos desde job_id {job_id}')
    finally:
        if conn:
            conn.close()

    result = None
    while result is None:
        try:
            # Conecta al servidor para obtener los resultados
            conn = socket.create_connection(('127.0.0.1', 1936))
            conn.send(b'\x01')  # Enviar código de operación para "get results"
            conn.send(job_id.to_bytes(4, 'little'))
            
            # Recibe y deserializa el resultado
            result_size = int.from_bytes(conn.recv(4), 'little')
            result = pickle.loads(conn.recv(result_size))
        finally:
            if conn:
                conn.close()
        sleep(1)  # Espera 1 segundo antes de intentar nuevamente
    print(f'El resultado es {result}')

if __name__ == '__main__':
    do_request(my_funs, 'Python rocks. Python es lo maximo'.split(' '))

