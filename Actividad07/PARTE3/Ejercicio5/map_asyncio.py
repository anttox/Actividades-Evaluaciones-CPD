import asyncio  # Importa el modulo asyncio para manejo de programación asincrona
import logging  
from collections import defaultdict  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Definimos la funcion de mapeo asincrona
async def map_function(word):
    try:
        return (word, 1)  # Devuelve una tupla con la palabra y el valor 1
    except Exception as e:
        logging.error(f"Error en la función de mapeo: {e}")  # Registra un mensaje de error si ocurre una excepcion
        return (word, 0)  # Devuelve la palabra y 0 en caso de error

# Definimos la funcion de reduccion asincrona
async def reduce_function(item):
    try:
        word, counts = item  # Desempaqueta la tupla en palabra y lista de cuentas
        return (word, sum(counts))  # Devuelve la palabra y la suma de las cuentas
    except Exception as e:
        logging.error(f"Error en la función de reducción: {e}")  # Registra un mensaje de error si ocurre una excepcion
        return (item[0], 0)  # Devuelve la palabra y 0 en caso de error

# Funcion asincrona para realizar MapReduce con seguimiento del progreso y manejo de excepciones
async def map_reduce_with_progress(my_input, mapper, reducer):
    distributor = defaultdict(list)  # Crea un defaultdict de listas para distribuir los resultados del mapeo

    logging.info("Inicio de la fase de mapeo")  # Registra el inicio de la fase de mapeo
    map_tasks = [mapper(word) for word in my_input]  # Crea una lista de tareas de mapeo
    map_results = await asyncio.gather(*map_tasks)  # Ejecuta las tareas de mapeo en paralelo y espera los resultados

    for word, count in map_results:
        distributor[word].append(count)  # Agrega el resultado al distribuidor

    logging.info("Inicio de la fase de reducción")  # Registra el inicio de la fase de reducción
    reduce_tasks = [reducer(item) for item in distributor.items()]  # Crea una lista de tareas de reducción
    results = await asyncio.gather(*reduce_tasks)  # Ejecuta las tareas de reducción en paralelo y espera los resultados

    logging.info("Proceso de MapReduce completado")  # Registra la finalizacion del proceso de MapReduce
    return results  # Devuelve los resultados

# Funcion para leer un archivo y preparar los datos de entrada
def read_file(file_path):
    with open(file_path, 'rt', encoding='utf-8') as file:
        words = filter(None, [word.strip().rstrip() for line in file for word in line.split()])  # Lee y procesa las palabras del archivo
    return words  # Devuelve la lista de palabras

# Funcion para preparar los datos de varios archivos
def prepare_data(file_paths):
    words = []  # Lista para almacenar todas las palabras
    for file_path in file_paths:
        words.extend(read_file(file_path))  # Agrega las palabras leidas de cada archivo a la lista
    return words  # Devuelve la lista completa de palabras

# Bloque principal del programa
if __name__ == "__main__":
    file_paths = ['texto1.txt', 'texto2.txt', 'texto3.txt']  # Lista de rutas de archivos a procesar
    words = prepare_data(file_paths)  # Prepara los datos leyendo los archivos
    
    results = asyncio.run(map_reduce_with_progress(words, map_function, reduce_function))  # Ejecuta el proceso de MapReduce de manera asincrona

    for word, count in sorted(results, key=lambda x: x[1], reverse=True):  # Ordena y muestra los resultados
        print(f"Palabra: '{word}', Frecuencia: {count}")  # Muestra cada palabra y su frecuencia
