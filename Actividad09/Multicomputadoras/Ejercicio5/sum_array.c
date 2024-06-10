#include <pthread.h>  // Para manejar el paralelismo utilizando pthreads
#include <stdio.h>    // Para manejar las funciones de entrada y salida estándar
#include <stdlib.h>   // Para manejar las funciones de utilidad general como rand()
#include <stdint.h>   // Para intptr_t

#define NUM_THREADS 4  // Número de hilos a utilizar
#define ARRAY_SIZE 1000000  // Tamaño del arreglo a sumar

int array[ARRAY_SIZE];  // Arreglo de tamaño 1,000,000
long long sum = 0;  // Variable global para almacenar la suma total
pthread_mutex_t mutex;  // Mutex para proteger la variable global sum

// Función para sumar un segmento del arreglo
void* sum_segment(void* arg) {
    int start = (intptr_t)arg * (ARRAY_SIZE / NUM_THREADS);  // Índice de inicio del segmento
    int end = start + (ARRAY_SIZE / NUM_THREADS);  // Índice de fin del segmento
    long long local_sum = 0;  // Variable local para almacenar la suma del segmento

    // Sumar los elementos del segmento
    for (int i = start; i < end; i++) {
        local_sum += array[i];
    }

    // Bloquear el mutex antes de actualizar la suma global
    pthread_mutex_lock(&mutex);
    sum += local_sum;  // Actualizar la suma global
    pthread_mutex_unlock(&mutex);  // Desbloquear el mutex

    return NULL;
}

int main() {
    pthread_t threads[NUM_THREADS];  // Arreglo de identificadores de hilos
    pthread_mutex_init(&mutex, NULL);  // Inicializar el mutex

    // Inicializar el arreglo con valores aleatorios
    for (int i = 0; i < ARRAY_SIZE; i++) {
        array[i] = rand() % 100;
    }

    // Crear los hilos para sumar los segmentos del arreglo
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, sum_segment, (void*)(intptr_t)i);
    }

    // Esperar a que todos los hilos terminen
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_mutex_destroy(&mutex);  // Destruir el mutex
    printf("Total sum: %lld\n", sum);  // Imprimir la suma total
    return 0;
}
