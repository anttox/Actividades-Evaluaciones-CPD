import concurrent.futures
import os # Podriamos usar os si queremos manejar archivos en implementacionees mas complejas

# Creamos una funcion para contar el numero de palabras de un archivo de texto
def count_words_in_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    word_count = len(text.split())
    return (file_path, word_count) # tupla con la ruta del archivo y el conteo de palabras

# Funcion para procesar una lista de archivos de texto en paralelo para contar las palabras en cada uno
def parallel_word_count(file_paths):
    """
    Procesa una lista de archivos de texto en paralelo para contar las palabras en cada uno.

    :param file_paths: Lista de rutas de archivos de texto.
    :return: Diccionario con las rutas de archivos y sus respectivos conteos de palabras.
    """
    results = {}
    # Usamos ThreadPoolExecutor para procesar los archivos en paralelo
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Se envian las tareas al executor gracias a submit
        future_to_file = {executor.submit(count_words_in_file, file_path): file_path for file_path in file_paths}
        # Recogemos los resultados conforme se vayan completando
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                file_path, count = future.result()
                results[file_path] = count
            except Exception as exc:
                print(f"{file_path} generó una excepción: {exc}")
    return results # Diccionario con las rutas de archivos y el conteo de palabras

# Lista de archivos de texto 
file_paths = ["file1.txt", "file2.txt", "file3.txt"]

# Se obtiene el conteo de palabras de los archivos de texto
word_counts = parallel_word_count(file_paths)

# Imprimir los resultados
print(word_counts)
