import easyocr
import string

#Iniciador de OCR
reader = easyocr.Reader(['en'], gpu=True)

# Diccionarios para la conversión de caracteres
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()

import string

import string

def formato_patentes(text):
    text = text.replace(" ", "").upper()  # Removemos espacios y convertimos a mayúsculas
    if len(text) == 6:  # Posible patente vieja
        if all(c in string.ascii_uppercase for c in text[:3]) and all(c.isdigit() for c in text[3:]):
            return True
    elif len(text) == 7:  # Posible patente nueva
        if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char) and \
           (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char) and \
           all((c.isdigit() or c in dict_char_to_int) for c in text[2:5]) and \
           (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char) and \
           (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char):
            return True
    return False


def validar_formato_patentes(text):
    text = text.replace(" ", "").upper()  # Removemos espacios y convertimos a mayúsculas
    if not formato_patentes(text):
        return False  # Retorna False si el formato no es válido

    if len(text) == 6:  # Para patentes viejas, no hay necesidad de conversión
        return text
    else:  # Para patentes nuevas, convertimos según los mapeos
        license_plate_ = ''
        # Usamos los diccionarios para la conversión si es necesario
        mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_char_to_int, 5: dict_int_to_char, 6: dict_int_to_char,
                   2: dict_char_to_int, 3: dict_char_to_int}
        for j, char in enumerate(text):
            if j in mapping and char in mapping[j]:
                license_plate_ += mapping[j][char]
            else:
                license_plate_ += char
        return license_plate_

def obtener_auto(patentes, vehiculos_track_id):
    x1, y1, x2, y2, puntuacion, class_id = patentes
    auto_marcado = False

    for i in range(len(vehiculos_track_id)):
        xauto1, yauto1, xauto2, yauto2, auto_id = vehiculos_track_id[i]
        
        if x1 > xauto1 and y1 > yauto1 and x2 > xauto2 and y2 > yauto2:
            auto_index = i
            auto_marcado = True
            break
    if auto_marcado:
        return vehiculos_track_id[auto_index]
    return -1, -1, -1, -1, -1

def leer_patente(patente_recortada):
    detecciones = reader.readtext(patente_recortada)

    for deteccion in detecciones:
        bbox, text, score = deteccion

        text = text.upper().replace(' ', '')
        if formato_patentes(text):
            return validar_formato_patentes(text), score
    return None, None