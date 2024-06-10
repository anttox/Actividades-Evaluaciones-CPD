class CacheLine:
    def __init__(self, data=None, state='Invalid'):
        self.data = data  # Dato almacenado en la linea de cache
        self.state = state  # Estado de la linea de cache

class Directory:
    def __init__(self):
        self.entries = {}  # Entradas del directorio que rastrean que procesadores tienen copias de que datos

    def update(self, address, processor_id):
        if address not in self.entries:
            self.entries[address] = set()  # Iniciamos el conjunto de procesadores para esta direccion
        self.entries[address].add(processor_id)  # Agregamos el procesador al conjunto

    def invalidate(self, address):
        if address in self.entries:
            self.entries[address] = set()  # Invalida todas las copias de esta direccion

class Processor:
    def __init__(self, id, directory):
        self.id = id  # Identificador del procesador
        self.cache = {}  # Cache local del procesador
        self.directory = directory  # Directorio compartido

    def read(self, address):
        if address in self.cache and self.cache[address].state != 'Invalid':
            return self.cache[address].data  # Devolvemos el dato si está en la cache y no esta invalido
        else:
            self.cache[address] = CacheLine()
            self.cache[address].state = 'Shared'
            self.directory.update(address, self.id)  # Actualizamos el directorio con el procesador actual
            self.cache[address].data = memory_read(address)  # Leemos el dato de la memoria principal
            return self.cache[address].data

    def write(self, address, data):
        if address in self.cache and (self.cache[address].state == 'Shared' or self.cache[address].state == 'Exclusive'):
            self.cache[address].state = 'Modified'
            self.cache[address].data = data
            self.directory.invalidate(address)  # Invalida todas las demas copias
        else:
            self.cache[address] = CacheLine()
            self.cache[address].state = 'Modified'
            self.cache[address].data = data
            self.directory.invalidate(address)  # Invalida todas las demas copias
            memory_write(address, data)  # Escribe el dato en la memoria principal

def memory_read(address):
    return "data_from_memory"  # Simula una lectura de la memoria principal

def memory_write(address, data):
    pass  # Simula una escritura en la memoria principal

# Ejemplo
directory = Directory()
p1 = Processor(1, directory)
p2 = Processor(2, directory)

# Simulacion de operaciones de lectura y escritura
p1.write(0x1A, "data1")
print(f"Processor 1 writes to 0x1A: {p1.cache[0x1A].data}, State: {p1.cache[0x1A].state}")
p2.read(0x1A)
print(f"Processor 2 reads from 0x1A: {p2.cache[0x1A].data}, State: {p2.cache[0x1A].state}")
print(directory.entries)
