// Comando para compilar el codigo: mpicc -o enviar_mensajes enviar_mensajes.c
// Comando para ejecutar el codigo: mpiexec -n 4 ./enviar_mensajes

#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);  // Iniciamos el entorno MPI

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Obtenemos el rango del proceso
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Obtenemos el tamaño del comunicador

    if (rank == 0) {
        char message[] = "Hola desde el rango 0";
        for (int i = 1; i < size; i++) {
            MPI_Send(message, strlen(message) + 1, MPI_CHAR, i, 0, MPI_COMM_WORLD);  // Se envia el mensaje
        }
    } else {
        char message[256];  // Asegurarse de que el buffer sea lo suficientemente grande para evitar truncamientos
        // MPI_Status status: Verifica el tamaño real del mensaje recibido y manejarlo adecuadamente
        MPI_Status status;
        MPI_Recv(message, 256, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);  // Se recibe el mensaje

        // Obtenemos el tamaño real del mensaje recibido
        int count;
        MPI_Get_count(&status, MPI_CHAR, &count);

        printf("Rango %d recibió mensaje: %s (tamaño: %d)\n", rank, message, count);
    }

    MPI_Finalize();  // Finalizamos el entorno MPI
    return 0;
}

