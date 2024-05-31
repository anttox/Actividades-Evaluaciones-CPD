#include <stdio.h>
#include <immintrin.h>

// Funcion para sumar dos arreglos de flotantes usando AVX
void add_float_arrays_avx(float *a, float *b, float *result, int n) {
    // Recorre los arreglos en pasos de 8 elementos
    for (int i = 0; i < n; i += 8) {
        // Carga 8 elementos del arreglo 'a' en un registro AVX
        __m256 va = _mm256_load_ps(&a[i]);
        // Carga 8 elementos del arreglo 'b' en un registro AVX
        __m256 vb = _mm256_load_ps(&b[i]);
        // Suma los dos registros AVX
        __m256 vr = _mm256_add_ps(va, vb);
        // Almacena el resultado en el arreglo 'result'
        _mm256_store_ps(&result[i], vr);
    }
}

int main() {
    // Iniciamos dos arreglos de flotantes con 8 elementos cada uno
    float a[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    float b[8] = {9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
    float result[8];

    // Llamamos a la funcion para sumar los arreglos
    add_float_arrays_avx(a, b, result, 8);

    // Imprimimos el resultado
    for (int i = 0; i < 8; i++) {
        printf("%f\n", result[i]);
    }

    return 0;
}

// Comando para correr el programa: gcc -o suma_avx suma_avx.c -mavx