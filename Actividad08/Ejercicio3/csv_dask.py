# Importamos Dask para procesar un conjunto de archivos CSV en paralelo
import dask.dataframe as dd

# Funcion para el procesamiento paralelo de archivos CSV
def parallel_csv_processing(file_paths):
    # Con read_csv leemos varios archivos CSV en un DataFrame de Dask
    df = dd.read_csv(file_paths)
    
    # Calculamos el promedio de la columna especifica ('target_column') en paralelo
    average_value = df['target_column'].mean().compute()
    
    # Devuelve el valor promedio calculado
    return average_value

# Lista de rutas de archivos CSV a procesar (Ejemplos) PARTE A MEJORAR
file_paths = ['Inventario.csv', 'Inventario2.csv']

# Llamamos a la funcion de procesamiento paralelo para obtener el promedio
average = parallel_csv_processing(file_paths)

# Imprime el valor promedio calculado
print(f"Average value: {average}")
