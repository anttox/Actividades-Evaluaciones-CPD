import dill as pickle
import socket
from time import sleep

# Funciones del cliente
def my_funs():
    def mapper(v):
        return v, 1

    def reducer(my_args):
        v, obs = my_args
        return v, sum(obs)
    return mapper, reducer

def get_node_loads(nodos):
    # loads = lista vacia para almacenar las cargas 
    loads = []
    for nodo, port in nodos:
        conn = None
        try:
            conn = socket.create_connection((nodo, port))
            conn.send(b'\x02')  # Código para obtener la carga
            load = int.from_bytes(conn.recv(4), 'little')
            loads.append(((nodo, port), load))
        finally:
            if conn:
                conn.close()
    return loads

def distribute_tasks(data, nodos):
    loads = get_node_loads(nodos)
    # Ordenamos los nodos por su carga
    loads.sort(key=lambda x: x[1])
    # Se crea un diccionario con cada nodo como clave y una lista vacia como valor
    assignments = {nodo: [] for nodo in nodos}
    
    for i, item in enumerate(data):
        # Selecciona el nodo con la menor carga
        nodo, _ = loads[i % len(loads)]
        assignments[nodo].append(item)
        # Actualizamos la carga del nodo
        loads[i % len(loads)] = (nodo, loads[i % len(loads)][1] + 1)

    return assignments

def do_request(my_funs, data):
    nodos = [('127.0.0.1', 1936), ('127.0.0.1', 1937), ('127.0.0.1', 1938)]  # Lista de nodos y puertos
    assignments = distribute_tasks(data, nodos)

    job_ids = []

    for (nodo, port), items in assignments.items():
        if items:
            conn = None
            try:
                conn = socket.create_connection((nodo, port))
                conn.send(b'\x00')
                my_code = pickle.dumps(my_funs.__code__)
                conn.send(len(my_code).to_bytes(4, 'little', signed=False))
                conn.send(my_code)
                my_data = pickle.dumps(items)
                conn.send(len(my_data).to_bytes(4, 'little'))
                conn.send(my_data)
                # Recibe el ID del trabajo
                job_id = int.from_bytes(conn.recv(4), 'little')
                job_ids.append(((nodo, port), job_id))
                # Imprimimos el ID del trabajo y el nodo
                print(f'Obteniendo datos desde job_id {job_id} en nodo {nodo}:{port}')
            finally:
                if conn:
                    conn.close()

    results = []
    # Bucle hasta que se reciban todos los resultados
    while len(results) < len(job_ids):
        for (nodo, port), job_id in job_ids:
            try:
                conn = socket.create_connection((nodo, port))
                conn.send(b'\x01')
                conn.send(job_id.to_bytes(4, 'little'))
                result_size = int.from_bytes(conn.recv(4), 'little')
                result = pickle.loads(conn.recv(result_size))
                if result is not None:
                    results.append(result)
            finally:
                if conn:
                    conn.close()
            sleep(1)
    
    final_result = sum(results)
    print(f'El resultado es {final_result}')

if __name__ == '__main__':
    do_request(my_funs, 'Python rocks. Python es lo maximo'.split(' '))
