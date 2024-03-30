import easyocr
import string
import pytesseract

#Iniciador de OCR
reader = easyocr.Reader(['en'], gpu=True)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def formato_patentes(text):
    text = text.replace(" ", "").upper()  # Removemos espacios y convertimos a mayúsculas
    if len(text) == 6:  #Patente vieja
        if all(c in string.ascii_uppercase for c in text[:3]) and all(c.isdigit() for c in text[3:]):
            return True
    elif len(text) == 7:  #Patente nueva
        if (text[0] in string.ascii_uppercase) and \
           (text[1] in string.ascii_uppercase) and \
           all((c.isdigit()) for c in text[2:5]) and \
           (text[5] in string.ascii_uppercase) and \
           (text[6] in string.ascii_uppercase):
            return True
    return False

def leer_patente_ocr(patente_recortada):
    detecciones = reader.readtext(patente_recortada)
    for deteccion in detecciones:
        bbox, text, score = deteccion
        text = text.upper().replace(' ', '')
        if len(text) in [6, 7] and formato_patentes(text):
            return text, score
    return None, None

def leer_patente_tesseract(patente_recortada):
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(patente_recortada, config=custom_config).replace("\n", "").replace("\f", "").replace(" ", "").strip()
    #print(text)
    if len(text) in [6, 7] and formato_patentes(text):
        return text, None  # PyTesseract no proporciona un score directamente
    return None, None

def obtener_auto(patentes, vehiculos_track_id):
    x1, y1, x2, y2, puntuacion, class_id = patentes
    auto_marcado = False

    for i in range(len(vehiculos_track_id)):
        xauto1, yauto1, xauto2, yauto2, auto_id = vehiculos_track_id[i]
        
        if xauto1 <= x1 <= xauto2 and yauto1 <= y1 <= yauto2 and \
           xauto1 <= x2 <= xauto2 and yauto1 <= y2 <= yauto2:
            auto_index = i
            auto_marcado = True
            break
            
    if auto_marcado:
        return vehiculos_track_id[auto_index]
    return -1, -1, -1, -1, -1

def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                    'license_plate_bbox', 'license_plate_bbox_score', 
                                                    'license_number', 'license_number_score',
                                                    'license_plate_tesseract', 'license_plate_tesseract_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    car_bbox = '[{} {} {} {}]'.format(results[frame_nmr][car_id]['car']['bbox'][0],
                                                        results[frame_nmr][car_id]['car']['bbox'][1],
                                                        results[frame_nmr][car_id]['car']['bbox'][2],
                                                        results[frame_nmr][car_id]['car']['bbox'][3])
                    license_plate_bbox = '[{} {} {} {}]'.format(results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                    results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                    results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                    results[frame_nmr][car_id]['license_plate']['bbox'][3])
                    license_plate_text = results[frame_nmr][car_id]['license_plate']['text']
                    license_plate_text_score = results[frame_nmr][car_id]['license_plate']['text_score']
                    
                    # Add license_plate_tesseract information
                    license_plate_tesseract = results[frame_nmr][car_id]['license_plate_tesseract']['text']
                    license_plate_tesseract_score = results[frame_nmr][car_id]['license_plate_tesseract']['text_score']

                    f.write('{},{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                                car_id,
                                                                car_bbox,
                                                                license_plate_bbox,
                                                                results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                                license_plate_text,
                                                                license_plate_text_score,
                                                                license_plate_tesseract,
                                                                license_plate_tesseract_score))
        f.close()