import torch
import torch.multiprocessing as mp

# Funcion de simulacion de Monte Carlo
def monte_carlo_simulation(num_simulations):
    count_inside = 0
    for _ in range(num_simulations):
        # Generar dos numeros aleatorios entre 0 y 1
        x, y = torch.rand(2)
        # Comprobar si el punto (x, y) esta dentro del circulo unitario
        if x**2 + y**2 <= 1.0:
            count_inside += 1
    return count_inside

# Funcion para ejecutar la simulacion de Monte Carlo en paralelo
def parallel_monte_carlo(num_simulations, num_processes):
    with mp.Pool(num_processes) as pool:
        # Dividir las simulaciones entre los procesos
        results = pool.map(monte_carlo_simulation, [num_simulations // num_processes] * num_processes)
    # Calcular la estimacion de Pi
    return sum(results) / num_simulations * 4

if __name__ == "__main__":
    num_simulations = 1000000  # Numero total de simulaciones
    num_processes = 4          # Numero de procesos para paralelizar
    # Ejecutar la simulacion en paralelo
    pi_estimate = parallel_monte_carlo(num_simulations, num_processes)
    # Imprimir el valor estimado de Pi
    print(f"El valor de Pi estimado: {pi_estimate}")
