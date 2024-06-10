from multiprocessing import Process, Manager

# Funcion para incrementar el valor de una clave en el diccionario compartido
def incrementar(diccionario_compartido, clave):
    for _ in range(1000):  # Incrementa el valor de la clave 1000 veces
        diccionario_compartido[clave] += 1

if __name__ == "__main__":
    manager = Manager()  # Creamos un Manager para manejar la memoria compartida entre procesos
    diccionario_compartido = manager.dict()  # Creamos un diccionario compartido
    diccionario_compartido["contador"] = 0  # Inicializamos la clave "contador" en 0

    procesos = []  # Lista para almacenar los procesos
    for _ in range(4):  # Creamos 4 procesos
        p = Process(target=incrementar, args=(diccionario_compartido, "contador"))
        procesos.append(p)  # Añadimos el proceso a la lista
        p.start()  # Iniciamos el proceso

    for p in procesos:  # Espera a que todos los procesos terminen
        p.join()

    # Imprimimos el valor final del contador compartido
    print(f"Valor final del contador: {diccionario_compartido['contador']}")

