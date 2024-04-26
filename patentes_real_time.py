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
    
    # Mostrar el fotograma actual
    cv2.imshow('Frame', frame)
    
    # Esperar la tecla 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
