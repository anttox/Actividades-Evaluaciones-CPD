import time
import numpy as np
import matplotlib.pyplot as plt

# Funcion para medir la latencia de acceso a datos en un array de tamaño especifico
def measure_latency(array_size):
    # Creamos un array de ceros con el tamaño especificado
    array = np.zeros(array_size)
    
    # Registramos el tiempo de inicio
    start_time = time.perf_counter()
    
    # Accedemos a todos los elementos del array para medir la latencia
    for i in range(array_size):
        array[i] += 1
    
    # Registramos el tiempo de finalizacion
    end_time = time.perf_counter()
    
    # Calculamos la latencia total
    latency = end_time - start_time
    return latency

# Definimos diferentes tamaños de array para la prueba
sizes = [10**3, 10**4, 10**5, 10**6, 10**7]
latencies = []

# Medimos la latencia para cada tamaño de array
for size in sizes:
    latency = measure_latency(size)
    latencies.append(latency)
    print(f"Tamaño del array: {size}, Latencia: {latency:.6f} segundos")

# Graficamos los resultados de las mediciones de latencia
plt.plot(sizes, latencies, marker='o')
plt.xlabel('Tamaño del Array')
plt.ylabel('Latencia (segundos)')
plt.xscale('log')
plt.yscale('log')
plt.title('Latencia de Acceso a Datos en Diferentes Tamaños de Array')
plt.show()
