from ultralytics import YOLO 
import cv2
from sort import *
from utils import obtener_auto, leer_patente, write_csv


resultados = {}
#Rastrear los vehículos
mot_tracker = Sort()

#Cargar modelos
coco_model = YOLO('./models/yolov8n.pt') #Modelo pre-entrenado de yolo
detector_patentes = YOLO("./models/best.pt")

#Cargar vídeo
cap = cv2.VideoCapture("videos/test.mp4")

vehiculos = [2, 3, 5 ,7]

#Leer frames
num_frame = -1
ret = True
while ret:
    num_frame += 1
    #Ubicación vídeos
    ret, frame = cap.read()
    if ret:
        resultados[num_frame] = {}
        #Detectar los vehículos
        detecciones = coco_model(frame)[0]
        detecciones_autos = []
        for deteccion in detecciones.boxes.data.tolist():
            x1, y1, x2, y2, puntuacion, class_id = deteccion
            if int(class_id) in vehiculos:
                detecciones_autos.append([x1, y1, x2, y2, puntuacion])
                #print (detecciones_autos)

        #Seguir los vehículos 
        tracks_id = mot_tracker.update(np.asarray(detecciones_autos))
        #print(tracks_id)

        #Detectar las patentes
        patentes = detector_patentes(frame)[0]
        for patente in patentes.boxes.data.tolist():
            x1, y1, x2, y2, puntuacion, class_id = patente
            #Asignar la patente a cada vehículo
            xauto1, yauto1, xauto2, yauto2, auto_id = obtener_auto(patente, tracks_id)

            if auto_id != -1:
                #Recortar la patente
                patente_recortada = frame[int(y1):int(y2), int(x1):int(x2), :] 

                #Aplicar filtros a la patente
                patente_recortada_gris = cv2.cvtColor(patente_recortada, cv2.COLOR_BGR2GRAY)
                _, patente_recortada_thresh = cv2.threshold(patente_recortada_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                bordes_canny = cv2.Canny(patente_recortada_thresh, 100, 200)
                _, patente_recortada_thresh2 = cv2.threshold(bordes_canny, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cv2.imshow('Bordes Canny', bordes_canny)
                cv2.imshow('thresh', patente_recortada_thresh2)
                cv2.waitKey(0)
                #Leer número de la patente
                patente_texto, patente_texto_score = leer_patente(patente_recortada_thresh)
                print (patente_texto)
                if patente_texto is not None:
                    resultados[num_frame][auto_id] = {'car': {'bbox': [xauto1, yauto1, xauto2, yauto2]},
                                                    'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': patente_texto,
                                                                        'bbox_score': puntuacion,
                                                                        'text_score': patente_texto_score}}


#Mostrar los resultados
write_csv(resultados, './test.csv')            


    