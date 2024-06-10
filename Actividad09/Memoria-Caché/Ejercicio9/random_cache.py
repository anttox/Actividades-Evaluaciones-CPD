import random # Usamos random para seleccionar aleatoriamente una clave para eliminar cuando la cache está llena

# Clase que implementa una cache con politica de reemplazo aleatorio
class RandomCache:
    def __init__(self, capacity):
        self.capacity = capacity  # Capacidad maxima de la cache
        self.cache = {}  # Diccionario para almacenar los elementos de la cache

    # Metodo para obtener un valor de la cache
    def get(self, key):
        return self.cache.get(key, -1)  # Retorna el valor asociado a la clave o -1 si la clave no esta en la cache

    # Metodo para insertar o actualizar un valor en la cache
    def put(self, key, value):
        if key not in self.cache:
            if len(self.cache) >= self.capacity:
                evict = random.choice(list(self.cache.keys()))  # Se selecciona aleatoriamente una clave para eliminar
                del self.cache[evict]  # Se elimina la clave seleccionada de la cache
        self.cache[key] = value  # Se inserta el nuevo valor en la cache

# Simulacion de acceso a la cache
cache = RandomCache(3)  # Creamos una cache con politica de reemplazo aleatorio y capacidad para 3 elementos
operations = [
    ("put", 1, "data1"), ("put", 2, "data2"), ("put", 3, "data3"), ("get", 1),
    ("put", 4, "data4"), ("get", 2), ("put", 5, "data5"), ("get", 3), ("get", 1)
]

# Ejecutamos las operaciones de la parte simulacion
for op in operations:
    if op[0] == "put":
        cache.put(op[1], op[2])  # Insertamos o actualizamos un valor en la cache
        print(f"Put: {op[1]} -> {op[2]}")
    elif op[0] == "get":
        result = cache.get(op[1])  # Obtenemos un valor de la cache
        print(f"Get: {op[1]} -> {result}")
    print(f"Estado de la caché: {list(cache.cache.items())}")  # Muestra el estado actual de la cache
