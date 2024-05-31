#include <stdio.h>
#include <xmmintrin.h> // Incluye las intrinsecas de SSE (Streaming SIMD Extensions) que permiten utilizar instrucciones SIMD

// Funcion para sumar dos arreglos de flotantes utilizando SIMD con instrucciones SSE
void add_float_arrays(float *a, float *b, float *result, int n) {
    // Recorre los arreglos en pasos de 4 elementos
    for (int i = 0; i < n; i += 4) {
        // Carga 4 elementos del arreglo 'a' en un registro SIMD __m128
        __m128 va = _mm_load_ps(&a[i]);
        // Carga 4 elementos del arreglo 'b' en un registro SIMD __m128
        __m128 vb = _mm_load_ps(&b[i]);
        // Suma los dos registros SIMD
        __m128 vr = _mm_add_ps(va, vb);
        // Almacena el resultado en el arreglo 'result'
        _mm_store_ps(&result[i], vr);
    }
}

int main() {
    // Iniciamos dos arreglos de flotantes con 4 elementos cada uno
    float a[4] = {1.0, 2.0, 3.0, 4.0};
    float b[4] = {5.0, 6.0, 7.0, 8.0};
    float result[4];

    // Llamamos a la funcion para sumar los arreglos
    add_float_arrays(a, b, result, 4);

    // Imprimimos el resultado
    for (int i = 0; i < 4; i++) {
        printf("%f\n", result[i]);
    }

    return 0;
}
