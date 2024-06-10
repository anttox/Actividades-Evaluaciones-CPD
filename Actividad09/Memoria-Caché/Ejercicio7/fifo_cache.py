from collections import deque

# Clase que implementa una cache FIFO (First In, First Out)
class FIFOCache:
    def __init__(self, capacity):
        self.capacity = capacity  # Capacidad maxima de la cache
        self.cache = {}  # Diccionario para almacenar los elementos de la cache
        self.order = deque()  # Cola para mantener el orden de insercion de los elementos

    # Metodo para obtener un valor de la cache
    def get(self, key):
        return self.cache.get(key, -1)  # Retorna el valor asociado a la clave o -1 si la clave no esta en la cache

    # Metodo para insertar o actualizar un valor en la cache
    def put(self, key, value):
        if key not in self.cache:
            if len(self.cache) >= self.capacity:
                oldest = self.order.popleft()  # Eliminamos el elemento mas antiguo de la cache
                del self.cache[oldest]  # Eliminamos el elemento del diccionario de la cache
            self.cache[key] = value  # Insertamos el nuevo valor en la cache
            self.order.append(key)  # Añadimos la clave a la cola para mantener el orden de insercion
        else:
            self.cache[key] = value  # Actualizamos el valor en la cache si la clave ya existe

# Simulacion de acceso a la cache
cache = FIFOCache(3)  # Crear una cache FIFO con capacidad para 3 elementos
operations = [
    ("put", 1, "data1"), ("put", 2, "data2"), ("put", 3, "data3"), ("get", 1),
    ("put", 4, "data4"), ("get", 2), ("put", 5, "data5"), ("get", 3), ("get", 1)
]

# Ejecutamos las operaciones de simulacion
for op in operations:
    if op[0] == "put":
        cache.put(op[1], op[2])  # Insertamos o actualizamos un valor en la cache
        print(f"Put: {op[1]} -> {op[2]}")
    elif op[0] == "get":
        result = cache.get(op[1])  # Obtenemos un valor de la cache
        print(f"Get: {op[1]} -> {result}")
    print(f"Estado de la caché: {list(cache.cache.items())}")  # Muestra el estado actual de la cache
