from collections import defaultdict, OrderedDict

# Clase que implementa una cache LFU (Least Frequently Used)
class LFUCache:
    def __init__(self, capacity):
        self.capacity = capacity  # Capacidad mqxima de la cache
        self.cache = {}  # Diccionario para almacenar los elementos de la cache
        self.freq = defaultdict(OrderedDict)  # Diccionario para mantener las frecuencias de uso
        self.min_freq = 0  # Minima frecuencia de uso

    # Metodo para obtener un valor de la cache
    def get(self, key):
        if key not in self.cache:
            return -1  # Retorna -1 si la clave no está en la cache
        freq = self.cache[key][1]  # Obtenemos la frecuencia actual de la clave
        self.cache[key][1] += 1  # Incrementamos la frecuencia de uso
        value = self.cache[key][0]  # Obtenemos el valor asociado a la clave
        del self.freq[freq][key]  # Elimina mosla clave del diccionario de frecuencia actual
        if not self.freq[freq]:
            del self.freq[freq]  # Eliminamos la entrada de frecuencia si esta vacia
            if self.min_freq == freq:
                self.min_freq += 1  # Actualizamos la minima frecuencia si es necesario
        self.freq[freq + 1][key] = None  # Añadimos la clave al nuevo nivel de frecuencia
        return value  # Retornamos el valor asociado a la clave

    # Metodo para insertar o actualizar un valor en la cache
    def put(self, key, value):
        if self.capacity == 0:
            return  # No hace nada si la capacidad es 0
        if key in self.cache:
            self.cache[key][0] = value  # Actualiza el valor si la clave ya existe
            self.get(key)  # Actualiza la frecuencia de uso
            return
        if len(self.cache) >= self.capacity:
            evict = next(iter(self.freq[self.min_freq]))  # Se encuentra la clave para eliminar
            del self.freq[self.min_freq][evict]  # Eliminamos la clave del diccionario de frecuencia minima
            del self.cache[evict]  # Eliminamos la clave del diccionario de la caché
        self.cache[key] = [value, 1]  # Insertamos el nuevo valor en la cache con frecuencia 1
        self.freq[1][key] = None  # Añadimos la clave al diccionario de frecuencia 1
        self.min_freq = 1  # Resetea la minima frecuencia a 1

# Simulacion de acceso a la cache
cache = LFUCache(3)  # Creamos una cache LFU con capacidad para 3 elementos
operations = [
    ("put", 1, "data1"), ("put", 2, "data2"), ("put", 3, "data3"), ("get", 1),
    ("put", 4, "data4"), ("get", 2), ("put", 5, "data5"), ("get", 3), ("get", 1)
]

# Ejecutamos las operaciones de simulacion
for op in operations:
    if op[0] == "put":
        cache.put(op[1], op[2])  # Inserta o actualiza un valor en la caché
        print(f"Put: {op[1]} -> {op[2]}")
    elif op[0] == "get":
        result = cache.get(op[1])  # Obtiene un valor de la caché
        print(f"Get: {op[1]} -> {result}")
    print(f"Estado de la caché: {list(cache.cache.items())}")  # Muestra el estado actual de la caché
