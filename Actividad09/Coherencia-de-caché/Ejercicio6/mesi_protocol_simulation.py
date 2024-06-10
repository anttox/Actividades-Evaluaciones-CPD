# Simulador del Protocolo MESI
class CacheLine:
    def __init__(self):
        self.state = 'Invalid'  # Estado inicial de la linea de cache
        self.data = None  # Dato almacenado en la linea de cache

class Processor:
    def __init__(self, id):
        self.id = id  # Identificador del procesador
        self.cache = {}  # Cache local del procesador

    def read(self, address):
        if address in self.cache and self.cache[address].state != 'Invalid':
            # Si la linea esta en la cache y no esta invalida, devolver el dato
            return self.cache[address].data
        else:
            # Si la linea no está en la cachi o está invalida, cargar de memoria principal
            self.cache[address] = CacheLine()
            self.cache[address].state = 'Shared'  # Cambiar el estado a 'Shared'
            self.cache[address].data = memory_read(address)  # Leer el dato de la memoria principal
            return self.cache[address].data

    def write(self, address, data):
        if address in self.cache and (self.cache[address].state == 'Shared' or self.cache[address].state == 'Exclusive'):
            # Si la linea está en estado 'Shared' o 'Exclusive', se modifica directamente
            self.cache[address].state = 'Modified'  # Cambiar el estado a 'Modified'
            self.cache[address].data = data
        else:
            # Si la linea no está en la cache o está invalida, cargar y modificar
            self.cache[address] = CacheLine()
            self.cache[address].state = 'Modified'  # Cambiar el estado a 'Modified'
            self.cache[address].data = data
            memory_write(address, data)  # Escribir el dato en la memoria principal

def memory_read(address):
    # Simula una lectura de memoria principal
    return "data_from_memory"

def memory_write(address, data):
    # Simula una escritura en memoria principal
    pass

# Ejemplo
p1 = Processor(1)
p2 = Processor(2)

# Simulacion de operaciones de escritura y lectura
p1.write(0x1A, "data1")
print(f"Processor 1 writes to 0x1A: {p1.cache[0x1A].data}, State: {p1.cache[0x1A].state}")
p2.read(0x1A)
print(f"Processor 2 reads from 0x1A: {p2.cache[0x1A].data}, State: {p2.cache[0x1A].state}")
