import cv2
import flet as ft
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
from sort import Sort
from combinarOCR import obtener_auto, leer_patente_ocr, leer_patente_tesseract, leer_patente_google, write_csv
import imutils

def main(page: ft.Page):
    page.title = "Detección de Vehículos y Patentes en Tiempo Real"
    page.update()
    
    def update_image(image_control, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        buf = BytesIO()
        pil_img.save(buf, format='PNG')
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        image_control.src_base64 = img_b64
        page.update()

    mot_tracker = Sort()
    coco_model = YOLO('./models/yolov8n.pt')
    detector_patentes = YOLO("./models/best.pt")
    vehiculos = [2, 3, 5, 7]
    threshold = 0.5
    resultados = {}

    cap = cv2.VideoCapture(2)
    width = int(1920)
    ASPECT_RATIO = 16 / 9
    height = int(width / ASPECT_RATIO)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    num_frame = 0

    image_control = ft.Image(width=640, height=480)
    page.add(image_control)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el fotograma")
            break

        detecciones = coco_model(frame)[0]
        detecciones_autos = [deteccion[:5] for deteccion in detecciones.boxes.data.tolist() if int(deteccion[5]) in vehiculos]
        tracks_id = mot_tracker.update(np.asarray(detecciones_autos))

        patentes = detector_patentes(frame)[0]
        for patente in patentes.boxes.data.tolist():
            x1, y1, x2, y2, puntuacion, class_id = patente
            if puntuacion > threshold:
                xauto1, yauto1, xauto2, yauto2, auto_id = obtener_auto(patente, tracks_id)

                patente_recortada = frame[int(y1):int(y2), int(x1):int(x2)]
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
                    if w >= 25 and h >= 100:
                        chars.append(contour)
                if chars:
                    chars = np.vstack([chars[i] for i in range(0, len(chars))])
                    hull = cv2.convexHull(chars)

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
        update_image(image_control, frame)
        num_frame += 1

    cap.release()

ft.app(target=main)