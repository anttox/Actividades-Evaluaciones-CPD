// Comnando para correr el codigo: gcc -o mutexes mutexes.c -pthread
// Usamoso la opcion -pthread para habilitar el uso de pthreads

#include <pthread.h>
#include <stdio.h>

int counter = 0;  // Variable global compartida entre los hilos
pthread_mutex_t lock;  // Declaramos un mutex para la sincronizacion

// Funcion que sera ejecutada por cada hilo
void* increment(void* arg) {
    for (int i = 0; i < 1000; i++) {
        pthread_mutex_lock(&lock);  // Bloqueamos el mutex antes de acceder al contador
        counter++;                  // Incrementamos el contador de forma segura
        pthread_mutex_unlock(&lock); // Desbloqueaos el mutex despues de incrementar el contador
    }
    return NULL;
}

int main() {
    pthread_t threads[4];  // Arreglo para almacenar los identificadores de los hilos
    pthread_mutex_init(&lock, NULL);  // Iniciamos el mutex antes de su uso

    // Creamos 4 hilos para ejecutar la funcion 'increment'
    for (int i = 0; i < 4; i++) {
        pthread_create(&threads[i], NULL, increment, NULL);
    }

    // Esperamos que los 4 hilos terminen su ejecucion
    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_mutex_destroy(&lock);  // Se destruye el mutex despues de su uso
    printf("Valor final del contador: %d\n", counter);  // Imprimimos el valor final del contador
    return 0;
}

