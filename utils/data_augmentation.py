import os, cv2
from dotenv import load_dotenv
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

load_dotenv()
ruta_imagenes = Path(os.getenv('DATASET_PATH'))
ruta_salida = Path(os.getenv('OUTPUT_PATH'))
ruta_salida.mkdir(parents=True, exist_ok=True)

# Configuración de las técnicas de Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,  
    width_shift_range=0.2,  
    height_shift_range=0.2,  
    zoom_range=0.3,  
    brightness_range=[0.2, 2],  
    shear_range=0.15,  
    horizontal_flip=True,  
    fill_mode='nearest'
)

# Función para añadir ruido gaussiano
def añadir_ruido_gaussiano(imagen, varianza=0.01):
    filas, columnas, canales = imagen.shape
    gauss = np.random.normal(0, varianza ** 0.5, (filas, columnas, canales)) * 255
    imagen_con_ruido = np.clip(imagen + gauss, 0, 255).astype('uint8')
    return imagen_con_ruido

# Función para añadir distorsiones aleatorias
def añadir_distorcion_aleatoria(imagen, factor_distorsion=0.05):
    alto, ancho = imagen.shape[:2]
    desplazamiento_aleatorio = np.random.randint(-int(alto * factor_distorsion), int(alto * factor_distorsion), size=(4, 2))
    puntos_src = np.float32([[0, 0], [ancho, 0], [0, alto], [ancho, alto]])
    puntos_dst = puntos_src + desplazamiento_aleatorio.astype(np.float32)
    matriz_transformacion = cv2.getPerspectiveTransform(puntos_src, puntos_dst)
    imagen_distorsionada = cv2.warpPerspective(imagen, matriz_transformacion, (ancho, alto), borderMode=cv2.BORDER_REFLECT)
    return imagen_distorsionada

# Procesar cada imagen en la carpeta
for ruta_imagen in ruta_imagenes.glob("*.[jp][pn]g"):  
    imagen = cv2.imread(str(ruta_imagen))
    
    if imagen is None:
        print(f"No se pudo cargar la imagen {ruta_imagen.name}. Saltando...")
        continue

    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)  
    # Expandir dimensiones para que coincidan con la entrada esperada por el generador
    imagen_expandidas = np.expand_dims(imagen_rgb, axis=0)
    # Generar imágenes aumentadas utilizando ImageDataGenerator
    iterador_aumento = datagen.flow(imagen_expandidas, batch_size=1)

    for i in range(10):  
        imagen_aumentada = next(iterador_aumento)[0].astype('uint8')

        # Añadir ruido gaussiano y distorsiones
        imagen_aumentada = añadir_ruido_gaussiano(imagen_aumentada)
        imagen_aumentada = añadir_distorcion_aleatoria(imagen_aumentada)
        # Guardar la imagen aumentada
        imagen_bgr = cv2.cvtColor(imagen_aumentada, cv2.COLOR_RGB2BGR)
        nombre_archivo_aumentado = ruta_salida / f"aumentada_{i}_{ruta_imagen.name}"
        cv2.imwrite(str(nombre_archivo_aumentado), imagen_bgr)

    print(f"Imágenes aumentadas generadas para {ruta_imagen.name}")

print("Proceso de aumento de datos completado.")