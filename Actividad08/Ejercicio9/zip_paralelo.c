#include <stdio.h>
#include <stdlib.h>
#include <pthread.h> # Para funciones de manejo de hilos
#include <zlib.h> # Para funciones de compresion y descompresion

// Estructura para pasar los datos a los hilos
typedef struct {
    char *input_file;
    char *output_file;
} ThreadData;

// Función que se ejecutaa en cada hilo para comprimir un archivo
void *compress_file(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    FILE *source = fopen(data->input_file, "rb");
    FILE *dest = fopen(data->output_file, "wb");

    if (!source || !dest) {
        perror("Error al abrir el archivo");
        pthread_exit(NULL);
    }

    char in_buffer[1024];
    char out_buffer[1024];
    z_stream stream = {0};
    deflateInit(&stream, Z_DEFAULT_COMPRESSION);

    int read;
    // Leemos el archivo fuente y comprimimos su contenido
    while ((read = fread(in_buffer, 1, sizeof(in_buffer), source)) > 0) {
        stream.avail_in = read;
        stream.next_in = (Bytef *)in_buffer;
        do {
            stream.avail_out = sizeof(out_buffer);
            stream.next_out = (Bytef *)out_buffer;
            deflate(&stream, Z_NO_FLUSH);
            fwrite(out_buffer, 1, sizeof(out_buffer) - stream.avail_out, dest);
        } while (stream.avail_out == 0);
    }

    // Finalizmmosla compresion
    do {
        stream.avail_out = sizeof(out_buffer);
        stream.next_out = (Bytef *)out_buffer;
        deflate(&stream, Z_FINISH);
        fwrite(out_buffer, 1, sizeof(out_buffer) - stream.avail_out, dest);
    } while (stream.avail_out == 0);

    deflateEnd(&stream);
    fclose(source);
    fclose(dest);
    pthread_exit(NULL);
}

int main() {
    // Lista de archivos de entrada y salida
    char *input_files[] = {"file1.txt", "file2.txt", "file3.txt"};
    char *output_files[] = {"file1.txt.gz", "file2.txt.gz", "file3.txt.gz"};
    int num_files = 3;

    // Declaramos los hilos y los datos para cada hilo
    pthread_t threads[num_files];
    ThreadData thread_data[num_files];

    // Se crea los hilos para comprimir los archivos
    for (int i = 0; i < num_files; i++) {
        thread_data[i].input_file = input_files[i];
        thread_data[i].output_file = output_files[i];
        pthread_create(&threads[i], NULL, compress_file, &thread_data[i]);
    }

    // Esperamos a que todos los hilos terminen con pthread_join
    for (int i = 0; i < num_files; i++) {
        pthread_join(threads[i], NULL);
    }

    return 0;
}
