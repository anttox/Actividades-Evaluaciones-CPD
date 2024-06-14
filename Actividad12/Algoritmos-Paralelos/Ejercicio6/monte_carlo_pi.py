import multiprocessing as mp
import random
import time

# Funcion que realiza parte del calculo de Monte Carlo para estimar Pi
# Genera n puntos aleatorios y cuenta cuantos caen dentro del circulo de radio 1
def monte_carlo_pi_part(n):
    count = 0
    for _ in range(n):
        x, y = random.random(), random.random()
        if x**2 + y**2 <= 1.0:
            count += 1
    return count

# Funcion que paraleliza el calculo de Monte Carlo para estimar Pi utilizando multiples procesos
# Divide el numero total de muestras entre los procesos disponibles
def parallel_monte_carlo_pi(total_samples, num_processes):
    pool = mp.Pool(processes=num_processes)
    samples_per_process = [total_samples // num_processes] * num_processes
    counts = pool.map(monte_carlo_pi_part, samples_per_process)
    pi_estimate = sum(counts) / total_samples * 4
    return pi_estimate

if __name__ == "__main__":
    total_samples = 10**7  # Numero total de muestras
    num_processes = 4  # Numero de procesos a utilizar

    # Ejecutamos el calculo de Pi en paralelo
    start_time = time.time()
    pi_estimate = parallel_monte_carlo_pi(total_samples, num_processes)
    end_time = time.time()

    print(f"Pi estimado: {pi_estimate}")
    print(f"Tiempo tomado: {end_time - start_time} en segundos")

    # Ejecutamos el calculo de Pi de manera secuencial para comparacion
    start_time_seq = time.time()
    pi_estimate_seq = monte_carlo_pi_part(total_samples) / total_samples * 4
    end_time_seq = time.time()

    print(f"Pi estimado (secuencial): {pi_estimate_seq}")
    print(f"Tiempo tomado (secuencial): {end_time_seq - start_time_seq} en segundos")
