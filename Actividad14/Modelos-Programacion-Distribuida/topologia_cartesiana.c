#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Función de operación personalizada para suma de cuadrados
void sum_of_squares(void *in, void *inout, int *len, MPI_Datatype *dptr) {
    int i;
    for (i = 0; i < *len; i++) {
        ((double*)inout)[i] += ((double*)in)[i] * ((double*)in)[i];
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);

    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    double local_value = rank + 1.0;
    double global_value;

    MPI_Op custom_op;
    MPI_Op_create(&sum_of_squares, 1, &custom_op);

    MPI_Reduce(&local_value, &global_value, 1, MPI_DOUBLE, custom_op, 0, cart_comm);

    if (rank == 0) {
        printf("Resultado de la suma de cuadrados: %f\n", global_value);
    }

    MPI_Op_free(&custom_op);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();

    return 0;
}

