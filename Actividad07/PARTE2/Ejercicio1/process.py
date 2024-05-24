from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import time
def map_reduce_ultra_naive(my_input, mapper, reducer):
    map_results = map(mapper, my_input)

    distributor = defaultdict(list)
    for key, value in map_results:
        distributor[key].append(value)

    return list(map(reducer, distributor.items()))

def map_reduce_processed(my_input, mapper, reducer):
    with ProcessPoolExecutor() as executor:
        map_futures = {executor.submit(mapper, item): item for item in my_input}
        map_results = []
        for future in as_completed(map_futures):
            map_results.append(future.result())

    distributor = defaultdict(list)
    for key, value in map_results:
        distributor[key].append(value)

    with ProcessPoolExecutor() as executor:
        reduce_futures = {executor.submit(reducer, item): item for item in distributor.items()}
        reduce_results = []
        for future in as_completed(reduce_futures):
            reduce_results.append(future.result())

    return reduce_results

# Función que toma una palabra y devuelve una tupla con la palabra y el número 1
def emiter(word):
    return (word, 1)

# Función que toma una tupla (palabra, lista de unos) y devuelve una tupla con la palabra y la suma de los unos, efectivamente contando cuántas veces aparece cada palabra
def counter(emitted):
    return (emitted[0], sum(emitted[1]))

# Esta linea divide la cadena en una lista de palabras = .split
words = 'Python es lo mejor Python rocks'.split(' ')

# Sincronico original
start_time = time.time()
sync_result = list(map_reduce_ultra_naive(words, emiter, counter))
sync_time = time.time() - start_time

# Paralelizacion con procesos
start_time = time.time()
process_result = map_reduce_processed(words, emiter, counter)
process_time = time.time() - start_time

print("Resultado sincronico:", sync_result)
print("Tiempo sincronico:", sync_time)
print("Resultado paralelizacion con procesos:", process_result)
print("Tiempo paralelizacion con procesos:", process_time)
