# Variable global para rastrear el trafico de coherencia
traffic = 0

class CacheLine:
    def __init__(self):
        self.state = 'Invalid'  # Estado inicial de la linea de cache
        self.data = None  # Dato almacenado en la linea de cache

class Processor:
    def __init__(self, id):
        self.id = id  # Identificador del procesador
        self.cache = {}  # Cachelocal del procesador

    def read(self, address):
        global traffic
        if address in self.cache and self.cache[address].state != 'Invalid':
            return self.cache[address].data  # Devuelve el dato si está en la cache y no está inválido
        else:
            self.cache[address] = CacheLine()
            self.cache[address].state = 'Shared'  # Cambia el estado a 'Shared'
            traffic += 1  # Incrementa el trafico de coherencia
            self.cache[address].data = memory_read(address)  # Lee el dato de la memoria principal
            return self.cache[address].data

    def write(self, address, data):
        global traffic
        if address in self.cache and (self.cache[address].state == 'Shared' or self.cache[address].state == 'Exclusive'):
            self.cache[address].state = 'Modified'  # Cambia el estado a 'Modified'
            self.cache[address].data = data  # Actualiza el dato en la cache
        else:
            self.cache[address] = CacheLine()
            self.cache[address].state = 'Modified'  # Cambia el estado a 'Modified'
            traffic += 1  # Incrementa el trafico de coherencia
            self.cache[address].data = data
            memory_write(address, data)  # Escribe el dato en la memoria principal

def memory_read(address):
    return "data_from_memory"  # Simula una lectura de la memoria principal

def memory_write(address, data):
    pass  # Simula una escritura en la memoria principal

# Ejemplo 
p1 = Processor(1)
p2 = Processor(2)

# Simulacion de operaciones de lectura y escritura
p1.write(0x1A, "data1")
p2.read(0x1A)
print(f"Traffic: {traffic} transactions")
