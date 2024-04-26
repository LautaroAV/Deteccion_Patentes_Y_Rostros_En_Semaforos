from ultralytics import YOLO
import cv2
import numpy as np
from sort import Sort
from combinarOCR import obtener_auto, leer_patente_ocr, leer_patente_tesseract, leer_patente_google, write_csv
import pandas as pd
import imutils

# Inicialización de rastreador y modelos
mot_tracker = Sort()
coco_model = YOLO('./models/yolov8n.pt')
detector_patentes = YOLO("./models/best.pt")

# Definiciones y variables
vehiculos = [2, 3, 5, 7]
threshold = 0.5
resultados = {}

# Captura de video desde la cámara
cap = cv2.VideoCapture(1)  # El argumento 0 indica la cámara predeterminada, puedes cambiarlo si tienes múltiples cámaras
cap.set(cv2.CAP_PROP_FPS, 30)  # Puedes cambiar el valor 30 a la velocidad de cuadros deseada

num_frame = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el fotograma")
        break
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
            gray = cv2.cvtColor(imagen_ampliada, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            contours = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)

            chars = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # Solo los contornos grandes perdurarán, ya que corresponden a los números que nos interesan.
                if w >= 25 and h >= 100:
                    chars.append(contour)
            if chars:
                chars = np.vstack([chars[i] for i in range(0, len(chars))])
                hull = cv2.convexHull(chars)

                # Creamos una máscara y la alargamos.
                mask = np.zeros(imagen_ampliada.shape[:2], dtype='uint8')
                cv2.drawContours(mask, [hull], -1, 255, -1)
                mask = cv2.dilate(mask, None, iterations=2)
                final = cv2.bitwise_and(opening, opening, mask=mask)

                patente_texto, patente_texto_score = leer_patente_ocr(final)
                patente_texto_tesseract, patente_texto_score_tesseract = leer_patente_tesseract(final)
                patente_texto_google, patente_texto_score_google = leer_patente_google(final)
                print("Patente EasyOCR: " + str(patente_texto))
                print("Patente Tesseract: " + str(patente_texto_tesseract))
                print("Patente Google: " + str(patente_texto_google))
                if patente_texto is not None:
                    if num_frame not in resultados:
                        resultados[num_frame] = {}
                    resultados[num_frame][auto_id] = {
                        'car': {'bbox': [xauto1, yauto1, xauto2, yauto2]},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': patente_texto,
                            'bbox_score': puntuacion,
                            'text_score': patente_texto_score
                        },
                        'license_plate_tesseract': {
                            'text': patente_texto_tesseract,
                            'text_score': patente_texto_score_tesseract
                        },
                        'license_plate_google': {
                            'text': patente_texto_google,
                            'text_score': patente_texto_score_google
                        }
                    }
    
    # Mostrar el fotograma actual
    cv2.imshow('Frame', frame)
    
    # Esperar la tecla 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    num_frame += 1  # Incrementar el número de fotograma

# Generar .csv con los resultados
csv_path = './data/resultados_video_en_tiempo_real.csv'
write_csv(resultados, csv_path)


def procesar_video_salida(resultados, video_sin_extension):
    df = pd.DataFrame.from_dict(resultados, orient='index')
    df = df.stack().apply(pd.Series)
    df.reset_index(inplace=True)
    df = df.rename(columns={'level_0': 'frame_nmr', 'level_1': 'car_id'})
    df[['xauto1', 'yauto1', 'xauto2', 'yauto2']] = pd.DataFrame(df['car'].tolist(), index=df.index)
    df[['x1', 'y1', 'x2', 'y2']] = pd.DataFrame(df['license_plate']['bbox'].tolist(), index=df.index)
    df.drop(columns=['car', 'license_plate'], inplace=True)

    # Calcular los valores más comunes en 'license_number', 'license_plate_tesseract', y 'license_plate_google' por 'car_id'
    license_number_counts = df.groupby(['car_id', 'license_plate'])['frame_nmr'].count().reset_index(name='license_number_count')
    tesseract_counts = df.groupby(['car_id', 'license_plate_tesseract'])['frame_nmr'].count().reset_index(name='tesseract_count')
    google_counts = df.groupby(['car_id', 'license_plate_google'])['frame_nmr'].count().reset_index(name='google_count')

    # Fusionar los recuentos de las placas
    merged_df = pd.concat([license_number_counts[['car_id', 'license_plate', 'license_number_count']], 
                           tesseract_counts[['car_id', 'license_plate_tesseract', 'tesseract_count']]
                           .rename(columns={'license_plate_tesseract': 'license_plate', 'tesseract_count': 'license_number_count'}),
                           google_counts[['car_id', 'license_plate_google', 'google_count']]
                           .rename(columns={'license_plate_google': 'license_plate', 'google_count': 'license_number_count'})])

    # Agrupar y sumar los recuentos de las placas
    merged_df = merged_df.groupby(['car_id', 'license_plate'], as_index=False)['license_number_count'].sum()

    # Ordenar el DataFrame resultante por 'car_id'
    merged_df = merged_df.sort_values(by='car_id')

    # Encontrar la patente más común por vehículo
    most_common_plate = merged_df.loc[merged_df.groupby('car_id')['license_number_count'].idxmax()]

    # Crear el diccionario car_license_mapping
    car_license_mapping = dict(zip(most_common_plate['car_id'], most_common_plate['license_plate']))

    # Cargar el video original
    cap = cv2.VideoCapture(video_sin_extension)
    video_path_out = video_sin_extension.replace('.mp4', '_out.mp4')
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
                car_bbox = [row['xauto1'], row['yauto1'], row['xauto2'], row['yauto2']]
                license_bbox = [row['x1'], row['y1'], row['x2'], row['y2']]

                # Dibujar rectángulos y texto en el frame
                cv2.rectangle(frame, (int(car_bbox[0]), int(car_bbox[1])), (int(car_bbox[2]), int(car_bbox[3])), (255, 0, 0), 2)
                cv2.rectangle(frame, (int(license_bbox[0]), int(license_bbox[1])), (int(license_bbox[2]), int(license_bbox[3])), (0, 255, 0), 4)
                cv2.putText(frame, license_plate_text, (int(license_bbox[0]), int(license_bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        else:
            # Si no hay información en el DataFrame para este frame, dibujar rectángulos azules para todos los autos
            for index, row in df[df['frame_nmr'] == frame_count].iterrows():
                car_bbox = [row['xauto1'], row['yauto1'], row['xauto2'], row['yauto2']]
                cv2.rectangle(frame, (int(car_bbox[0]), int(car_bbox[1])), (int(car_bbox[2]), int(car_bbox[3])), (255, 0, 0), 2)

        # Escribimos el frame en el video de salida
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Llamar a la función después de haber obtenido todos los resultados en tiempo real
procesar_video_salida(resultados, 'nombre_del_video.mp4')
