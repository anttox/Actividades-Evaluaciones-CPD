# Clase Node para un nodo de cache distribuida con consistencia eventual
class Node:
    def __init__(self, id):
        self.id = id  # Identificador del nodo
        self.data = {}  # Diccionario para almacenar los datos
        self.log = []  # Registro de operaciones para la correcta replicacion

    def put(self, key, value):
        # Almacenamos el valor en un diccionario de datos y registramos la operacion
        self.data[key] = value
        self.log.append((key, value))

    def get(self, key):
        # Se obtiene el valor asociado a una clave
        return self.data.get(key, None)

    def replicate(self, other_node):
        # Replicamos los datos y el registro de operaciones en otro nodo
        for entry in self.log:
            other_node.data[entry[0]] = entry[1]
        other_node.log.extend(self.log)

# Ejemplo 
if __name__ == "__main__":
    # Crear dos nodos
    node1 = Node('node1')
    node2 = Node('node2')

    # Almacenamos datos en el primer nodo
    node1.put('key1', 'value1')
    node1.put('key2', 'value2')

    # Replicamos los datos del nodo1 al nodo2
    node1.replicate(node2)

    # Obtenemos y mostramos los datos replicados en el segundo nodo
    print(f"Node 2 - Key 1: {node2.get('key1')}")  # Esperado: value1 para el node1
    print(f"Node 2 - Key 2: {node2.get('key2')}")  # Esperado: value2 para el node2
