import os, random, shutil
from dotenv import load_dotenv

load_dotenv()

dir_imagenes_origen = os.getenv('SOURCE_IMAGES_DIR')
dir_labels_origen = os.getenv('SOURCE_LABELS_DIR')

dir_imagenes_entrenamiento = os.getenv('TRAIN_IMAGES_DIR')
dir_labels_entrenamiento = os.getenv('TRAIN_LABELS_DIR')

dir_imagenes_validacion = os.getenv('VAL_IMAGES_DIR')
dir_labels_validacion = os.getenv('VAL_LABELS_DIR')

for dir_ruta in [dir_imagenes_entrenamiento, dir_labels_entrenamiento, dir_imagenes_validacion, dir_labels_validacion]:
    os.makedirs(dir_ruta, exist_ok=True)

# Obtener todas las imágenes en el directorio original
todas_imagenes = [f for f in os.listdir(dir_imagenes_origen) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Mezclar aleatoriamente las imágenes para una distribución aleatoria
random.shuffle(todas_imagenes)

# Calcular el número de imágenes para entrenamiento (80%) y el resto para validación (20%)
total_entrenamiento = int(len(todas_imagenes) * 0.80)

# Dividir imágenes en entrenamiento y validación
imagenes_entrenamiento, imagenes_validacion = todas_imagenes[:total_entrenamiento], todas_imagenes[total_entrenamiento:]

def mover_archivos(lista_imagenes, dir_img_origen, dir_lbl_origen, dir_img_destino, dir_lbl_destino):
    for imagen in lista_imagenes:
        nombre_imagen = os.path.splitext(imagen)[0] 
        ruta_img_origen = os.path.join(dir_img_origen, imagen)
        ruta_lbl_origen = os.path.join(dir_lbl_origen, f"{nombre_imagen}.txt")
        ruta_img_destino = os.path.join(dir_img_destino, imagen)
        ruta_lbl_destino = os.path.join(dir_lbl_destino, f"{nombre_imagen}.txt")
        
        if os.path.exists(ruta_img_origen):
            shutil.move(ruta_img_origen, ruta_img_destino)
        else:
            print(f"Imagen no encontrada: {ruta_img_origen}")
        
        if os.path.exists(ruta_lbl_origen):
            shutil.move(ruta_lbl_origen, ruta_lbl_destino)
        else:
            print(f"Label no encontrado para: {nombre_imagen}")

mover_archivos(imagenes_entrenamiento, dir_imagenes_origen, dir_labels_origen, dir_imagenes_entrenamiento, dir_labels_entrenamiento)
mover_archivos(imagenes_validacion, dir_imagenes_origen, dir_labels_origen, dir_imagenes_validacion, dir_labels_validacion)

print(f'Movidas {len(imagenes_entrenamiento)} imágenes y sus labels a las carpetas de entrenamiento.')
print(f'Movidas {len(imagenes_validacion)} imágenes y sus labels a las carpetas de validación.')