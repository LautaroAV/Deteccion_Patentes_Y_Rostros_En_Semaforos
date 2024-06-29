import cv2
import pandas as pd
import numpy as np
from ocr_detection import detect_license_plate, obtener_auto
from face_detection import detect_faces_in_frame

VEHICULOS = [2, 3, 5, 7]
THRESHOLD = 0.5

def process_video_frames(video_path, mot_tracker, coco_model, detector_patentes, face_detector, rostros_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = -1
    resultados = {}

    while cap.isOpened():
        frame_count += 1
        ret, frame = cap.read()
        if not ret:
            break

        resultados[frame_count] = {}

        # Detección de vehículos y seguimiento
        detecciones = coco_model(frame)[0]
        detecciones_autos = [
            deteccion[:5] for deteccion in detecciones.boxes.data.tolist()
            if int(deteccion[5]) in VEHICULOS
        ]

        if not detecciones_autos:
            continue

        tracks_id = mot_tracker.update(np.asarray(detecciones_autos))

        # Detección de patentes
        patentes = detector_patentes(frame)[0]
        for patente in patentes.boxes.data.tolist():
            x1, y1, x2, y2, puntuacion, _ = patente
            if puntuacion > THRESHOLD:
                xauto1, yauto1, xauto2, yauto2, auto_id = obtener_auto(patente, tracks_id)
                if auto_id == -1:
                    continue

                resultado_patente = detect_license_plate(frame, patente)
                if resultado_patente:
                    resultados[frame_count][auto_id] = {
                        'car': {'bbox': [xauto1, yauto1, xauto2, yauto2]},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': resultado_patente['text'],
                            'bbox_score': puntuacion,
                            'text_score': resultado_patente['text_score']
                        },
                        'tesseract': {
                            'text': resultado_patente['tesseract']['text'],
                            'text_score': resultado_patente['tesseract']['text_score']
                        },
                        'google': {
                            'text': resultado_patente['google']['text'],
                            'text_score': resultado_patente['google']['text_score']
                        }
                    }

        # Detección de rostros
        detect_faces_in_frame(frame, frame_count, resultados, face_detector, rostros_path)

    cap.release()
    return resultados

def generate_output_video(video_path, csv_path, resultados):
    df = pd.read_csv(csv_path, usecols=lambda col: 'faces' in col or col in [
        'frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score',
        'license_number', 'license_number_score', 'license_plate_tesseract', 'license_plate_google'
    ])

    merged_df = pd.concat([
        df.groupby(['car_id', 'license_number']).size().reset_index(name='license_number_count'),
        df.groupby(['car_id', 'license_plate_tesseract']).size().reset_index(name='tesseract_count').rename(columns={
            'license_plate_tesseract': 'license_number', 'tesseract_count': 'license_number_count'
        }),
        df.groupby(['car_id', 'license_plate_google']).size().reset_index(name='google_count').rename(columns={
            'license_plate_google': 'license_number', 'google_count': 'license_number_count'
        })
    ]).groupby(['car_id', 'license_number'], as_index=False)['license_number_count'].sum().sort_values(by='car_id')

    most_common_plate = merged_df.loc[merged_df.groupby('car_id')['license_number_count'].idxmax()]
    car_license_mapping = dict(zip(most_common_plate['car_id'], most_common_plate['license_number']))

    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter(
        video_path.replace('.mp4', '_out.mp4'),
        cv2.VideoWriter_fourcc(*'MP4V'),
        int(cap.get(cv2.CAP_PROP_FPS)),
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count in df['frame_nmr'].values:
            frame_info = df[df['frame_nmr'] == frame_count]
            for _, row in frame_info.iterrows():
                car_id = row['car_id']
                license_plate_text = car_license_mapping.get(car_id, "")
                car_bbox = [int(float(coord)) for coord in row['car_bbox'].strip('[]').split()]
                license_bbox = [int(float(coord)) for coord in row['license_plate_bbox'].strip('[]').split()]

                cv2.rectangle(frame, (car_bbox[0], car_bbox[1]), (car_bbox[2], car_bbox[3]), (255, 0, 0), 2)
                cv2.rectangle(frame, (license_bbox[0], license_bbox[1]), (license_bbox[2], license_bbox[3]), (0, 255, 0), 4)
                cv2.putText(frame, license_plate_text, (license_bbox[0], license_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                if 'faces' in row and pd.notna(row['faces']):
                    for face_path in row['faces'].strip('[]').replace("'", "").split(", "):
                        face = cv2.imread(face_path)
                        h, w, _ = face.shape
                        frame[0:h, 0:w] = face
                        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 2)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
