// Comando para compilar el codigo MPI: mpicc -o monte_carlo_pi monte_carlo_pi.c
// Comando para ejecutar el codigo: mpirun -np 4 ./monte_carlo_pi
// Aquí, -np 4 indica que el programa se ejecutará con 4 procesos. Ajusta el número de procesos según los recursos disponibles en tu sistema.

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    // Iniciamos el entorno MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    // Obtenemos el identificador del proceso actual (rango)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Obtenemos el numero total de procesos
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Definimos el numero total de muestras
    long long num_samples = 1000000;
    // Calculamos el numero de muestras locales para cada proceso
    long long local_samples = num_samples / size;
    // Inicializamos el contador local
    long long local_count = 0;
    // Semilla para el generador de numeros aleatorios basada en el rango del proceso
    unsigned int seed = rank;

    // Realizamos la simulacion de Monte Carlo en paralelo
    for (long long i = 0; i < local_samples; i++) {
        double x = (double)rand_r(&seed) / RAND_MAX;
        double y = (double)rand_r(&seed) / RAND_MAX;
        if (x * x + y * y <= 1.0) {
            local_count++;
        }
    }

    // Variable para almacenar el conteo total
    long long total_count;
    // Reducimos los conteos locales a un conteo total usando MPI_Reduce
    MPI_Reduce(&local_count, &total_count, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // El proceso con rango 0 calcula el valor de Pi
    if (rank == 0) {
        double pi_estimate = (4.0 * total_count) / num_samples;
        printf("Estimated Pi: %f\n", pi_estimate);
    }

    // Finalizamos el entorno MPI
    MPI_Finalize();
    return 0;
}
