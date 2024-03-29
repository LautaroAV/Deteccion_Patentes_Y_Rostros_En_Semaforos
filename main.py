from ultralytics import YOLO 
import cv2
import numpy as np
from sort import Sort
from easy_ocr import obtener_auto, leer_patente, write_csv
import pandas as pd

# Inicialización de rastreador y modelos
mot_tracker = Sort()
coco_model = YOLO('./models/yolov8n.pt')
detector_patentes = YOLO("./models/best.pt")

#Definiciones y variables
vehiculos = [2, 3, 5, 7]
threshold = 0.5
resultados = {}
nombre_video = "a2.mp4"
seleccionar_video = "videos/" + nombre_video

# Cargar vídeo
cap = cv2.VideoCapture(seleccionar_video)

#Primera Parte: Detección, seguimiento y generación de .csv
#Procesamiento de frames
num_frame = -1
ret = True
while ret:
    num_frame += 1
    ret, frame = cap.read()
    if ret:
        resultados[num_frame] = {}

        # Detección de vehículos y seguimiento
        detecciones = coco_model(frame)[0]
        detecciones_autos = [deteccion[:5] for deteccion in detecciones.boxes.data.tolist() if int(deteccion[5]) in vehiculos]
        tracks_id = mot_tracker.update(np.asarray(detecciones_autos))
        
        # Detección de patentes
        patentes = detector_patentes(frame)[0]
        for patente in patentes.boxes.data.tolist():
            x1, y1, x2, y2, puntuacion, class_id = patente
            if puntuacion > threshold:
                xauto1, yauto1, xauto2, yauto2, auto_id = obtener_auto(patente, tracks_id)

                # Recortar patente
                patente_recortada = frame[int(y1):int(y2), int(x1):int(x2)]

                # Redimensionar la imagen
                nuevo_ancho = 4 * patente_recortada.shape[1]
                nuevo_alto = 4 * patente_recortada.shape[0]
                imagen_ampliada = cv2.resize(patente_recortada, (nuevo_ancho, nuevo_alto))

                #Aplicar filtros a la patente
                patente_recortada_gris = cv2.cvtColor(imagen_ampliada, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(patente_recortada_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Encontrar contornos en la imagen umbralizada
                contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contorno_patente = max(contornos, key=cv2.contourArea)

                # Crear una máscara para la patente
                mascara = np.zeros_like(patente_recortada_gris)
                cv2.drawContours(mascara, [contorno_patente], -1, (255), thickness=cv2.FILLED)

                # Aplicar la máscara a la imagen umbralizada
                patente_final = cv2.bitwise_and(thresh, thresh, mask=mascara)
                # cv2.imshow('Bordes Canny', patente_final)
                # cv2.waitKey(0)

                patente_texto, patente_texto_score = leer_patente(patente_final)
                print(patente_texto)

                if patente_texto is not None:
                    resultados[num_frame][auto_id] = {
                        'car': {'bbox': [xauto1, yauto1, xauto2, yauto2]},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': patente_texto,
                            'bbox_score': puntuacion,
                            'text_score': patente_texto_score
                        }
                    }

# Generar .csv con los resultados
video_sin_extension = nombre_video.split('.')[0]
csv_path = './data/{}.csv'.format(video_sin_extension)
write_csv(resultados, csv_path)
cap.release()

# Segunda Parte: Procesamiento del vídeo de salida utilizando .csv
csv_path = './data/{}.csv'.format(video_sin_extension)
df = pd.read_csv(csv_path, usecols=['frame_nmr', 'car_id', 'car_bbox',
                                     'license_plate_bbox', 'license_plate_bbox_score',
                                     'license_number', 'license_number_score'])


patente_mas_comun_por_vehiculo = df.groupby('car_id')['license_number'].agg(lambda x: x.value_counts().idxmax()).reset_index()
car_license_mapping = dict(zip(patente_mas_comun_por_vehiculo['car_id'], patente_mas_comun_por_vehiculo['license_number']))

# VideoCapture y el VideoWriter
video_path = seleccionar_video
cap = cv2.VideoCapture(video_path)
video_path_out = video_path.replace('.mp4', '_out.mp4')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Verificamos si hay información en el DataFrame para este frame
    if frame_count in df['frame_nmr'].values:
        # Obtenemos la información para este frame
        frame_info = df[df['frame_nmr'] == frame_count]

        # Iteramos sobre cada registro en este frame
        for index, row in frame_info.iterrows():
            car_id = row['car_id']
            license_plate_text = car_license_mapping[car_id]

            # Convertir las coordenadas de la cadena a una lista
            car_bbox_str = row['car_bbox'].strip('[]').split()
            car_bbox = [float(coord) for coord in car_bbox_str]
            license_bbox_str = row['license_plate_bbox'].strip('[]').split()
            license_bbox = [float(coord) for coord in license_bbox_str]

            # Dibujar rectángulos y texto en el frame
            cv2.rectangle(frame, (int(car_bbox[0]), int(car_bbox[1])), (int(car_bbox[2]), int(car_bbox[3])), (255, 0, 0), 2)
            cv2.rectangle(frame, (int(license_bbox[0]), int(license_bbox[1])), (int(license_bbox[2]), int(license_bbox[3])), (0, 255, 0), 4)
            cv2.putText(frame, license_plate_text, (int(license_bbox[0]), int(license_bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    else:
        # Si no hay información en el DataFrame para este frame, dibujar rectángulos azules para todos los autos
        for index, row in df[df['frame_nmr'] == frame_count].iterrows():
            car_bbox_str = row['car_bbox'].strip('[]').split()
            car_bbox = [int(float(coord)) for coord in car_bbox_str]
            cv2.rectangle(frame, (car_bbox[0], car_bbox[1]), (car_bbox[2], car_bbox[3]), (255, 0, 0), 2)

    # Escribimos el frame en el video de salida
    out.write(frame)
    frame_count += 1

# Liberar los recursos
cap.release()
out.release()
cv2.destroyAllWindows()