# Clase DistributedCache para una cache distribuida con replicacion de datos
class DistributedCache:
    def __init__(self):
        self.nodes = {}  # Diccionario para almacenar los nodos de la cache

    def add_node(self, node_id):
        # Se agrega un nuevo nodo a la cache distribuida
        self.nodes[node_id] = {}

    def put(self, key, value):
        # Se replica el valor en todos los nodos de la cache
        for node in self.nodes.values():
            node[key] = value

    def get(self, node_id, key):
        # Obtenemos el valor asociado a una clave desde un nodo especificado
        return self.nodes[node_id].get(key, None)

# Ejemplo
if __name__ == "__main__":
    cache = DistributedCache()
    cache.add_node('node1')
    cache.add_node('node2')

    cache.put('key1', 'value1')
    print(f"Node 1: {cache.get('node1', 'key1')}")  # Esperado: value1 para node1
    print(f"Node 2: {cache.get('node2', 'key1')}")  # Esperado: value1 para node2
