from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import time
def map_reduce_ultra_naive(my_input, mapper, reducer):
    map_results = map(mapper, my_input)

    distributor = defaultdict(list)
    for key, value in map_results:
        distributor[key].append(value)

    return list(map(reducer, distributor.items()))

def map_reduce_threaded(my_input, mapper, reducer):
    with ThreadPoolExecutor() as executor:
        # submit = ejecuta funciones de forrma asincrona y devuelve objetos future
        map_futures = {executor.submit(mapper, item): item for item in my_input}
        map_results = []
        # as_completed = manejar las tareas conforme apenas se completen
        for future in as_completed(map_futures):
            map_results.append(future.result())

    distributor = defaultdict(list)
    for key, value in map_results:
        distributor[key].append(value)

    with ThreadPoolExecutor() as executor:
        reduce_futures = {executor.submit(reducer, item): item for item in distributor.items()}
        reduce_results = []
        for future in as_completed(reduce_futures):
            reduce_results.append(future.result())

    return reduce_results

# Esta linea divide la cadena en una lista de palabras = .split
words = 'Python es lo mejor Python rocks'.split(' ')

# Función lambda que toma una palabra y devuelve una tupla con la palabra y el número 1
# Etapa map
emiter = lambda word: (word, 1)
# Función lambda que toma una tupla (palabra, lista de unos) y devuelve una tupla con la palabra y la suma de los unos, efectivamente contando cuántas veces aparece cada palabra
# Etapa reduce
counter = lambda emitted: (emitted[0], sum(emitted[1]))

# Sincronico original
start_time = time.time()
sync_result = list(map_reduce_ultra_naive(words, emiter, counter))
sync_time = time.time() - start_time

# Paralelizacion con hilos
start_time = time.time()
thread_result = map_reduce_threaded(words, emiter, counter)
thread_time = time.time() - start_time

print("Resultado sincronico:", sync_result)
print("Tiempo sincronico:", sync_time)
print("Resultado paralelizacion con hilos:", thread_result)
print("Tiempo paralelizacion con hilos:", thread_time)
