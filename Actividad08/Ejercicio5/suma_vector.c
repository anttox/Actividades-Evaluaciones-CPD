#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_THREADS 4       // Número de hilos
#define VECTOR_SIZE 1000000 // Tamaño del vector

// Estructura para almacenar los datos del hilo
typedef struct {
    int start;      // Índice de inicio del segmento
    int end;        // Índice de fin del segmento
    double *vector; // Puntero al vector
    double sum;     // Suma parcial calculada por el hilo
} ThreadData;

// Función que cada hilo ejecutará para calcular la suma parcial
void *partial_sum(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    data->sum = 0.0;
    // Calcular la suma parcial del segmento asignado
    for (int i = data->start; i < data->end; i++) {
        data->sum += data->vector[i];
    }
    pthread_exit(NULL);
}

int main() {
    // Asignar memoria para el vector
    double *vector = (double *)malloc(VECTOR_SIZE * sizeof(double));
    // Inicializar el vector con valores aleatorios
    for (int i = 0; i < VECTOR_SIZE; i++) {
        vector[i] = rand() % 100;
    }

    pthread_t threads[NUM_THREADS];          // Array de identificadores de hilos
    ThreadData thread_data[NUM_THREADS];     // Array de datos de hilos
    int segment_size = VECTOR_SIZE / NUM_THREADS; // Tamaño del segmento asignado a cada hilo

    // Crear los hilos y asignar los segmentos del vector
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].start = i * segment_size;
        thread_data[i].end = (i == NUM_THREADS - 1) ? VECTOR_SIZE : (i + 1) * segment_size;
        thread_data[i].vector = vector;
        pthread_create(&threads[i], NULL, partial_sum, (void *)&thread_data[i]);
    }

    double total_sum = 0.0;
    // Esperar a que todos los hilos terminen y combinar los resultados
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
        total_sum += thread_data[i].sum;
    }

    // Imprimir la suma total
    printf("Total sum: %f\n", total_sum);
    // Liberar la memoria asignada para el vector
    free(vector);
    return 0;
}
