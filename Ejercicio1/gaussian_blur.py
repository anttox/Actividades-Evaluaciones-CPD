import cv2 # Cargamos la imagen
import numpy as np
from multiprocessing import Pool

# Funcion que aplica el desenfoque a la imagen
def aplicar_desenfoque(segmento):
    return cv2.GaussianBlur(segmento, (15, 15), 0)

def procesamiento_paralelo_imagen(ruta_imagen):
    # Cargar la imagen desde el archivo con imread
    imagen = cv2.imread(ruta_imagen)
    
    # Obtenenemos las dimensiones de la imagen
    altura, ancho = imagen.shape[:2]
    
    # Divididimos la imagen en 4 segmentos verticales
    segmentos = np.array_split(imagen, 4, axis=1)
    
    # Creamos un pool de procesos para el procesamiento paralelo
    with Pool(processes=4) as pool:
        # Aplicamos el filtro de desenfoque a cada segmento en paralelo
        segmentos_desenfocados = pool.map(aplicar_desenfoque, segmentos)
    
    # Unir los segmentos procesados en una sola imagen
    imagen_desenfocada = np.hstack(segmentos_desenfocados)
    
    # Guardar la imagen resultante en un archivo con imwrite
    cv2.imwrite('imagen_desenfocada.jpg', imagen_desenfocada)

# Llamar a la funcion principal con la ruta de la imagen de entrada
procesamiento_paralelo_imagen('gato.jpg')
