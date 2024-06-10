from collections import OrderedDict

# Clase que implementa una cache LRU (Least Recently Used)
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity  # Capacidad maxima de la cache
        self.cache = OrderedDict()  # Almacenammos de la cache usando OrderedDict

    # Metodo para obtener un valor de la cache
    def get(self, key):
        if key not in self.cache:
            return -1  # Retorna -1 si la clave no está en la cache
        self.cache.move_to_end(key)  # Mueve la clave al final para indicar que fue usada recientemente
        return self.cache[key]  # Retorna el valor asociado a la clave

    # Metodo para insertar o actualizar un valor en la cache
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)  # Mueve la clave al final si ya está en la cache
        self.cache[key] = value  # Inserta o actualiza el valor en la cache
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # Elimina el elemento menos recientemente usado si la capacidad se excede

# Simulacion de acceso a la cache
cache = LRUCache(5)  # Crear una caché LRU con capacidad para 5 elementos
operations = [
    ("put", 1, "data1"), ("put", 2, "data2"), ("get", 1), ("put", 3, "data3"),
    ("put", 4, "data4"), ("put", 5, "data5"), ("get", 2), ("put", 6, "data6"),
    ("get", 3), ("get", 1)
]

# Ejecutamos las operaciones de simulacion
for op in operations:
    if op[0] == "put":
        cache.put(op[1], op[2])  # Inserta o actualiza un valor en la cache
        print(f"Put: {op[1]} -> {op[2]}")
    elif op[0] == "get":
        result = cache.get(op[1])  # Obtiene un valor de la cache
        print(f"Get: {op[1]} -> {result}")
    print(f"Estado de la caché: {list(cache.cache.items())}")  # Muestra el estado actual de la cache
