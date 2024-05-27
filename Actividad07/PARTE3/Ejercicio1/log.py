# logging = monitoreo, depuración y mantenimiento de aplicaciones (errores)
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor as Executor, as_completed

# Configuracion del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Definimos las funciones de mapeo y reducción
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

# Función para realizar MapReduce con seguimiento del progreso y manejo de excepciones
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

# Preparamos los datos de entrada leyendo desde un archivo
def prepare_data(file_path):
    with open(file_path, 'rt', encoding='utf-8') as file:
        words = filter(None, [word.strip().rstrip() for line in file for word in line.split()])
    return words

# Ejecutamos el proceso de MapReduce
if __name__ == "__main__":
    words = prepare_data('texto.txt')
    results = map_reduce_with_progress(words, map_function, reduce_function)

    for word, count in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"Palabra: '{word}', Frecuencia: {count}")
