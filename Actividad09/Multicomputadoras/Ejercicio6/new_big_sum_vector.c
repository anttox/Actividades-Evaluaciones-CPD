// Comando para compilar el programa: mpicc -o new_big_sum_vector new_big_sum_vector.c
// Comando para ejecutar el programa: mpirun -np 4 ./new_big_sum_vector

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

    printf("Proceso %d iniciado de %d procesos\n", rank, size);

    // Definimos el tamaño del vector
    int vector_size = 1000000;
    // Calculamos el tamaño local del vector para cada proceso
    int local_size = vector_size / size;

    // Asignamos memoria para las porciones locales de los vectores
    double* local_A = (double*)malloc(local_size * sizeof(double));
    double* local_B = (double*)malloc(local_size * sizeof(double));
    double* local_C = (double*)malloc(local_size * sizeof(double));

    // Verificamos la asignación de memoria
    if (local_A == NULL || local_B == NULL || local_C == NULL) {
        printf("Proceso %d: Error en la asignación de memoria\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Iniciamos los vectores locales A y B con valores específicos
    for (int i = 0; i < local_size; i++) {
        local_A[i] = rank + 1;  // Ejemplo A, puede ser cualquier valor
        local_B[i] = rank + 2;  // Ejemplo B, puede ser cualquier valor
    }

    printf("Proceso %d: Vectores locales inicializados\n", rank);

    // Sumamos los vectores locales A y B y almacenamos el resultado en local_C
    for (int i = 0; i < local_size; i++) {
        local_C[i] = local_A[i] + local_B[i];
    }

    printf("Proceso %d: Suma de vectores locales completada\n", rank);

    // Imprimir los primeros 10 elementos de cada proceso
    printf("Resultados del proceso %d:\n", rank);
    for (int i = 0; i < 10; i++) {
        printf("local_C[%d] = %f\n", i, local_C[i]);
    }

    // Liberamos la memoria asignada
    free(local_A);
    free(local_B);
    free(local_C);

    // Finalizamos el entorno MPI
    MPI_Finalize();
    return 0;
}
