from multiprocessing import Pool

# Funcion para sumar todos los elementos de un segmento de la lista
def sum_segment(segment):
    return sum(segment)

if __name__ == "__main__":
    data = list(range(1000000))  # Lista de un millon de elementos
    num_processes = 4  # Numero de procesos 
    segment_size = len(data) // num_processes  # Tamaño de cada segmento

    # Divide la lista en segmentos, uno para cada proceso
    segments = [data[i * segment_size:(i + 1) * segment_size] for i in range(num_processes)]

    # Crea un Pool de procesos
    with Pool(num_processes) as pool:
        results = pool.map(sum_segment, segments)  # Asigna cada segmento a un proceso y se ejecuta

    total_sum = sum(results)  # Suma los resultados de todos los procesos
    print(f"Suma total: {total_sum}")
