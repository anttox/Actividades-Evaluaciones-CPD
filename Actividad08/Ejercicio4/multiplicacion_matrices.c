#include <stdio.h>  // Incluir las funciones de entrada/salida estandar
#include <stdlib.h> // Incluir las funciones de utilidad general
#include <omp.h>    // Incluir las definiciones y directivas de OpenMP

// Funcion para realizar la multiplicacion de matrices en paralelo
void multiplicacion_matrices_paralela(int N) {
    int i, j, k;
    // Asignamos memoria para las matrices A, B y C
    double **A = (double **)malloc(N * sizeof(double *));
    double **B = (double **)malloc(N * sizeof(double *));
    double **C = (double **)malloc(N * sizeof(double *));
    
    // Iniciamos las matrices A y B con valores aleatorios, y C con ceros
    for (i = 0; i < N; i++) {
        A[i] = (double *)malloc(N * sizeof(double));
        B[i] = (double *)malloc(N * sizeof(double));
        C[i] = (double *)malloc(N * sizeof(double));
        for (j = 0; j < N; j++) {
            A[i][j] = rand() % 100; // Asignamos valores aleatorios a la matriz A
            B[i][j] = rand() % 100; // Asignamos valores aleatorios a la matriz B
            C[i][j] = 0.0;          // Inicializamos la matriz C con ceros
        }
    }

    // Paralelizamos la multiplicacion de matrices utilizando OpenMP
    #pragma omp parallel for private(i, j, k) shared(A, B, C)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j]; // Realizar la multiplicación de matrices
            }
        }
    }

    // Imprimimos una pequeña parte de la matriz resultante para verificación (observar el programa)
    for (i = 0; i < 5; i++) {
        for (j = 0; j < 5; j++) {
            printf("%f ", C[i][j]);
        }
        printf("\n");
    }

    // Liberar la memoria asignada para las matrices A, B y C
    for (i = 0; i < N; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);
}

int main() {
    int N = 1000; // Tamaño de las matrices
    multiplicacion_matrices_paralela(N); // Aqui se da la multiplicacion de matrices en paralelo
    return 0;
}
