# Realiza pruebas de rendimiento variando el número de workers en ThreadPoolExecutor para observar cómo afecta al tiempo total de procesamiento del MapReduce.
import time  # Importa el módulo time para medir el tiempo de ejecución
import logging  # Importa el módulo logging para registrar eventos
from collections import defaultdict  # Importa defaultdict del módulo collections
from concurrent.futures import ThreadPoolExecutor as Executor, as_completed  # Importa Executor y as_completed del módulo concurrent.futures

# Configuración del logging para mostrar mensajes de INFO o superiores y un formato específico
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Definimos la función de mapeo
def map_function(word):
    try:
        return (word, 1)  # Devuelve una tupla con la palabra y el valor 1
    except Exception as e:
        logging.error(f"Error en la función de mapeo: {e}")  # Registra un mensaje de error si ocurre una excepción
        return (word, 0)  # Devuelve la palabra y 0 en caso de error

# Definimos la función de reducción
def reduce_function(item):
    try:
        word, counts = item  # Desempaqueta la tupla en palabra y lista de cuentas
        return (word, sum(counts))  # Devuelve la palabra y la suma de las cuentas
    except Exception as e:
        logging.error(f"Error en la función de reducción: {e}")  # Registra un mensaje de error si ocurre una excepción
        return (item[0], 0)  # Devuelve la palabra y 0 en caso de error

# Función para realizar MapReduce con seguimiento del progreso y manejo de excepciones
def map_reduce_with_progress(my_input, mapper, reducer, num_workers):
    distributor = defaultdict(list)  # Crea un defaultdict de listas para distribuir los resultados del mapeo

    with Executor(max_workers=num_workers) as executor:
        logging.info("Inicio de la fase de mapeo")  # Registra el inicio de la fase de mapeo
        future_to_word = {executor.submit(mapper, word): word for word in my_input}  # Envía tareas de mapeo a los workers
        
        for future in as_completed(future_to_word):
            try:
                word, count = future.result()  # Obtiene el resultado de la tarea de mapeo
                distributor[word].append(count)  # Agrega el resultado al distribuidor
            except Exception as e:
                logging.error(f"Error al obtener resultado del mapeo: {e}")  # Registra un mensaje de error si ocurre una excepción

        logging.info("Inicio de la fase de reducción")  # Registra el inicio de la fase de reducción
        results = list(executor.map(reducer, distributor.items()))  # Envía tareas de reducción a los workers
    
    logging.info("Proceso de MapReduce completado")  # Registra la finalización del proceso de MapReduce
    return results  # Devuelve los resultados

# Función para leer un archivo y preparar los datos de entrada
def read_file(file_path):
    with open(file_path, 'rt', encoding='utf-8') as file:
        words = filter(None, [word.strip().rstrip() for line in file for word in line.split()])  # Lee y procesa las palabras del archivo
    return words  # Devuelve la lista de palabras

# Función para preparar los datos de varios archivos
def prepare_data(file_paths):
    words = []  # Lista para almacenar todas las palabras
    with Executor() as executor:
        futures = {executor.submit(read_file, file_path): file_path for file_path in file_paths}  # Envía tareas de lectura de archivos a los workers
        for future in as_completed(futures):
            try:
                words.extend(future.result())  # Agrega las palabras leídas a la lista
            except Exception as e:
                logging.error(f"Error al leer el archivo {futures[future]}: {e}")  # Registra un mensaje de error si ocurre una excepción
    return words  # Devuelve la lista completa de palabras

# Bloque principal del programa
if __name__ == "__main__":
    file_paths = ['texto1.txt', 'texto2.txt', 'texto3.txt']  # Lista de rutas de archivos a procesar
    words = prepare_data(file_paths)  # Prepara los datos leyendo los archivos
    
    for workers in [1, 2, 4, 8, 16]:  # Prueba con diferentes números de workers
        start_time = time.time()  # Registra el tiempo de inicio
        results = map_reduce_with_progress(words, map_function, reduce_function, workers)  # Ejecuta el proceso de MapReduce
        end_time = time.time()  # Registra el tiempo de finalización
        print(f"Tiempo de ejecución con {workers} workers: {end_time - start_time:.2f} segundos")  # Muestra el tiempo de ejecución

    for word, count in sorted(results, key=lambda x: x[1], reverse=True):  # Ordena y muestra los resultados
        print(f"Palabra: '{word}', Frecuencia: {count}")  # Muestra cada palabra y su frecuencia
