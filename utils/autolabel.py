import cv2, shutil, os
from pathlib import Path
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()
ruta_imagenes = Path(os.getenv('TRAIN_PATH'))
ruta_base = Path(os.getenv('BASE_PATH'))
ruta_imagenes_base = ruta_base / 'images'
ruta_imagenes_detectadas = ruta_imagenes_base / 'images_detectadas'
ruta_etiquetas = ruta_base / 'label'
ruta_no_detecciones = ruta_base / 'no_detection'

for ruta in [ruta_imagenes_base, ruta_imagenes_detectadas, ruta_etiquetas, ruta_no_detecciones]:
    ruta.mkdir(parents=True, exist_ok=True)

# Cargar el modelo YOLO
modelo = YOLO('./models/patentesv3.pt')

# Convertir las coordenadas a formato YOLO
def convertir_a_formato_yolo(ancho_imagen, alto_imagen, caja):
    x1, y1, x2, y2 = caja
    x_centro = (x1 + x2) / 2.0 / ancho_imagen
    y_centro = (y1 + y2) / 2.0 / alto_imagen
    ancho = (x2 - x1) / ancho_imagen
    alto = (y2 - y1) / alto_imagen
    return x_centro, y_centro, ancho, alto

# Dibujar cuadros delimitadores en la imagen
def dibujar_cuadros_en_imagen(imagen, cajas):
    for caja in cajas:
        x1, y1, x2, y2 = map(int, caja)
        cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2) 
    return imagen

# Detección de patentes y generación de TXT en formato YOLO
def detectar_patentes_y_generar_txt(ruta_imagen, modelo):
    imagen = cv2.imread(str(ruta_imagen))
    
    if imagen is None:
        print(f"No se pudo cargar la imagen {ruta_imagen}. Saltando...")
        return False

    alto_imagen, ancho_imagen = imagen.shape[:2]
    resultados = modelo(imagen)
    cajas = [caja[:4].tolist() for resultado in resultados for caja in resultado.boxes.xyxy.cpu().numpy()]

    if not cajas:
        print(f'No se detectó ninguna patente en la imagen: {ruta_imagen}')
        shutil.copy(ruta_imagen, ruta_no_detecciones / ruta_imagen.name)
        return False

    shutil.copy(ruta_imagen, ruta_imagenes_base / ruta_imagen.name)

    nombre_txt = ruta_etiquetas / f"{ruta_imagen.stem}.txt"
    with open(nombre_txt, 'w') as archivo_salida:
        for caja in cajas:
            archivo_salida.write(f"0 {' '.join(map(str, convertir_a_formato_yolo(ancho_imagen, alto_imagen, caja)))}\n")

    cv2.imwrite(str(ruta_imagenes_detectadas / ruta_imagen.name), dibujar_cuadros_en_imagen(imagen.copy(), cajas))
    print(f'Imagen y TXT guardados para: {ruta_imagen}')
    return True

def procesar_ruta(ruta_imagenes, modelo):
    extensiones_imagen = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    for extension in extensiones_imagen:
        for ruta_imagen in ruta_imagenes.glob(extension):
            print(f'Procesando imagen: {ruta_imagen}')
            detectar_patentes_y_generar_txt(ruta_imagen, modelo)

procesar_ruta(ruta_imagenes, modelo)