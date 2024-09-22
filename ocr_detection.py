import os, io, cv2, easyocr, pytesseract, string, imutils
import numpy as np
from ultralytics import YOLO
from google.cloud import vision
from PIL import Image
from dotenv import load_dotenv 

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Inicialización de OCR
reader = easyocr.Reader(['en'], gpu=True)
tesseract_cmd_path = os.getenv('TESSERACT_CMD')
if not tesseract_cmd_path:
    raise Exception("Error: la variable de entorno TESSERACT_CMD no está definida o no se ha cargado correctamente.")

pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path

def initialize_ocr_models():
    coco_model = YOLO('./models/vehiculos.pt')
    detector_patentes = YOLO('./models/best.pt')
    return coco_model, detector_patentes

def formato_patentes(text):
    text = text.replace(" ", "").upper()
    if len(text) == 6:  # Patente vieja
        return all(c in string.ascii_uppercase for c in text[:3]) and all(c.isdigit() for c in text[3:])
    elif len(text) == 7:  # Patente nueva
        return (
            text[0] in string.ascii_uppercase and
            text[1] in string.ascii_uppercase and
            all(c.isdigit() for c in text[2:5]) and
            text[5] in string.ascii_uppercase and
            text[6] in string.ascii_uppercase
        )
    return False

def leer_patente_ocr(patente_recortada):
    detecciones = reader.readtext(patente_recortada)
    for _, text, score in detecciones:
        text = text.upper().replace(' ', '')
        if len(text) in [6, 7] and formato_patentes(text):
            return text, score
    return None, None

def leer_patente_tesseract(patente_recortada):
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789Ñ'
    text = pytesseract.image_to_string(patente_recortada, config=custom_config).replace("\n", "").replace("\f", "").replace(" ", "").strip()
    if len(text) in [6, 7] and formato_patentes(text):
        return text, None
    return None, None

def leer_patente_google(patente_recortada):
    google_credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not google_credentials_path:
        raise Exception("Error: la variable de entorno GOOGLE_APPLICATION_CREDENTIALS no está definida o no se ha cargado correctamente.")

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_credentials_path
    client = vision.ImageAnnotatorClient()
    image_pillow = Image.fromarray(patente_recortada)

    if not os.path.exists('./images/patentes'):
        os.makedirs('./images/patentes')

    image_pillow.save('./images/patentes/patente_temporal.jpg')

    with io.open('./images/patentes/patente_temporal.jpg', 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    full_text_annotation = response.full_text_annotation

    new_text = '' 
    confidence_score = None

    if texts:
        new_text = ''.join(texts[0].description.split())
        if full_text_annotation and full_text_annotation.pages:
            detected_languages = full_text_annotation.pages[0].property.detected_languages
            if detected_languages:
                confidence_score = detected_languages[0].confidence

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    if len(new_text) in [6, 7] and formato_patentes(new_text):
        return new_text, confidence_score
    else:
        return None, None

def detect_license_plate(frame, patente):
    x1, y1, x2, y2, puntuacion, _ = patente
    patente_recortada = frame[int(y1):int(y2), int(x1):int(x2)]
    imagen_ampliada = cv2.resize(patente_recortada, (0, 0), fx=4, fy=4)
    gray = cv2.cvtColor(imagen_ampliada, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    contours = imutils.grab_contours(cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))

    chars = [
        contour for contour in contours
        if cv2.boundingRect(contour)[2] >= 25 and cv2.boundingRect(contour)[3] >= 100
    ]

    if chars:
        hull = cv2.convexHull(np.vstack(chars))
        mask = cv2.dilate(cv2.drawContours(np.zeros(imagen_ampliada.shape[:2], dtype='uint8'), [hull], -1, 255, -1), None, iterations=2)
        final = cv2.bitwise_and(opening, opening, mask=mask)

        patente_texto, patente_texto_score = leer_patente_ocr(final)
        patente_texto_tesseract, patente_texto_score_tesseract = leer_patente_tesseract(final)
        patente_texto_google, patente_texto_score_google = leer_patente_google(final)
        
        print("Patente OCR " + str(patente_texto))
        print("Patente Tesseract " + str(patente_texto_tesseract))
        print("Patente Google " + str(patente_texto_google))

        if patente_texto:
            return {
                'bbox': [x1, y1, x2, y2],
                'text': patente_texto,
                'bbox_score': puntuacion,
                'text_score': patente_texto_score,
                'tesseract': {'text': patente_texto_tesseract, 'text_score': patente_texto_score_tesseract},
                'google': {'text': patente_texto_google, 'text_score': patente_texto_score_google}
            }
    return None


def obtener_auto(patentes, vehiculos_track_id):
    x1, y1, x2, y2, _, _ = patentes
    for vehiculo in vehiculos_track_id:
        xauto1, yauto1, xauto2, yauto2, auto_id = vehiculo
        if xauto1 <= x1 <= xauto2 and yauto1 <= y1 <= yauto2 and xauto1 <= x2 <= xauto2 and yauto1 <= y2 <= yauto2:
            return vehiculo
    return -1, -1, -1, -1, -1
