from multiprocessing import Process, Queue
import time
import random

# Ambas funciones sirven para el paralelismo de tareas
# Funcion del productor que produce numeros aleatorios y los pone en la cola
def producer(queue):
    for _ in range(10):
        item = random.randint(1, 100)
        queue.put(item)
        print(f"Produced {item}")
        time.sleep(random.random())  # Simula tiempo de produccion

# Funcion del consumidor que consume numeros de la cola
def consumer(queue):
    while True:
        item = queue.get()
        if item is None:  # Valor centinela para indicar el fin del procesamiento
            break
        print(f"Consumed {item}")
        time.sleep(random.random())  # Simula el tiempo de consumo

if __name__ == "__main__":
    queue = Queue()  # Cola para la comunicacion entre procesos

    # Creacion de procesos productores
    producers = [Process(target=producer, args=(queue,)) for _ in range(2)]
    # Creacion de procesos consumidores
    consumers = [Process(target=consumer, args=(queue,)) for _ in range(2)]
    
    # Etapa de sincronizacion
    # Inicio de procesos productores
    for p in producers:
        p.start()

    # Inicio de procesos consumidores
    for c in consumers:
        c.start()

    # Espera a que los procesos productores terminen
    for p in producers:
        p.join()

    # Añade valores centinela a la cola para indicar a los consumidores que terminen
    for _ in consumers:
        queue.put(None)

    # Espera a que los procesos consumidores terminen
    for c in consumers:
        c.join()
