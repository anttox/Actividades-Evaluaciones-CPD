# Ejercicio2: Modifica la función prepare_data para que pueda leer de múltiples archivos en paralelo usando ThreadPoolExecutor.
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor as Executor, as_completed
# logging.INFO = Nivel de severidad minimo que se registraran de los mensajes
# %(asctime)s = Muestra la fecha y hora en que se registro el mensaje
# %(levelname)s = Muestra el nivel de severidad del mensaje (INFO, WARNING, ERROR, etc.).
# %(message)s = Muestra el mensaje de registro
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def map_function(word):
    try:
        return (word, 1)
    except Exception as e:
        logging.error(f"Error en la función de mapeo: {e}")
        return (word, 0)

def reduce_function(item):
    try:
        word, counts = item
        return (word, sum(counts))
    except Exception as e:
        logging.error(f"Error en la función de reducción: {e}")
        return (item[0], 0)

def map_reduce_with_progress(my_input, mapper, reducer):
    distributor = defaultdict(list)

    with Executor() as executor:
        logging.info("Inicio de la fase de mapeo")
        future_to_word = {executor.submit(mapper, word): word for word in my_input}
        
        for future in as_completed(future_to_word):
            try:
                word, count = future.result()
                distributor[word].append(count)
            except Exception as e:
                logging.error(f"Error al obtener resultado del mapeo: {e}")

        logging.info("Inicio de la fase de reducción")
        results = list(executor.map(reducer, distributor.items()))
    
    logging.info("Proceso de MapReduce completado")
    return results

def read_file(file_path):
    with open(file_path, 'rt', encoding='utf-8') as file:
        words = filter(None, [word.strip().rstrip() for line in file for word in line.split()])
    return words

def prepare_data(file_paths):
    words = []
    with Executor() as executor:
        futures = {executor.submit(read_file, file_path): file_path for file_path in file_paths}
        for future in as_completed(futures):
            try:
                words.extend(future.result())
            except Exception as e:
                logging.error(f"Error al leer el archivo {futures[future]}: {e}")
    return words

if __name__ == "__main__":
    file_paths = ['texto1.txt', 'texto2.txt', 'texto3.txt']
    words = prepare_data(file_paths)
    results = map_reduce_with_progress(words, map_function, reduce_function)

    for word, count in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"Palabra: '{word}', Frecuencia: {count}")
