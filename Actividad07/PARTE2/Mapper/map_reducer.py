from collections import defaultdict
# Codigo de Thiago Rodriguez
# Operacion sicronica = sin cliente ni sleep
# my_input = datos de entrada
# mapper = genera tuplas (clave, valor)
# defauldict = agrupar todos los valores asociados con la misma clave (etapa reduce)
# reducer = aplica reduce a cada par (clave, sum(valor))
def map_reduce_ultra_naive(my_input, mapper, reducer):
    map_results = map(mapper, my_input)

    distributor = defaultdict(list)
    for key, value in map_results:
        distributor[key].append(value)

    return map(reducer, distributor.items())

# Esta linea divide la cadena en una lista de palabras = .split
words = 'Python es lo mejor Python rocks'.split(' ')

# Función lambda que toma una palabra y devuelve una tupla con la palabra y el número  1
# Etapa map
emiter = lambda word: (word, 1)
# Función lambda que toma una tupla (palabra, lista de unos) y devuelve una tupla con la palabra y la suma de los unos, efectivamente contando cuántas veces aparece cada palabra
# Etapa reduce
counter = lambda emitted: (emitted[0], sum(emitted[1]))

a = list(map_reduce_ultra_naive(words, emiter, counter))

print(a)
