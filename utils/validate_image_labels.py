import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
ruta_imagenes = Path(os.getenv('IMAGES_PATH'))
ruta_etiquetas = Path(os.getenv('LABELS_PATH'))

extensiones_imagenes = ['.jpg', '.jpeg', '.png', '.bmp']

# Listar todas las imágenes en la carpeta de imágenes
imagenes = [archivo.stem for archivo in ruta_imagenes.iterdir() if archivo.suffix.lower() in extensiones_imagenes]
# Listar todos los archivos .txt en la carpeta de etiquetas
etiquetas = [archivo.stem for archivo in ruta_etiquetas.iterdir() if archivo.suffix.lower() == '.txt']

# Comparar y encontrar imágenes sin archivo .txt correspondiente
imagenes_sin_txt = [imagen for imagen in imagenes if imagen not in etiquetas]

if not imagenes_sin_txt:
    print("Todas las imágenes tienen su archivo .txt correspondiente.")
else:
    print("Las siguientes imágenes no tienen archivo .txt correspondiente:")
    for imagen in imagenes_sin_txt:
        print(imagen)
