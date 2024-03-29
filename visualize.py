import os
import pandas as pd
import cv2
from ultralytics import YOLO

VIDEOS_DIR = os.path.join('.', 'videos')

video_path = os.path.join(VIDEOS_DIR, 'test.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

# Asumiendo que el CSV está en la misma carpeta que este script
csv_path = os.path.join('.', 'test.csv')

# Leer el archivo CSV
df = pd.read_csv(csv_path)

# Convertir DataFrame a diccionario para un acceso más rápido
license_info = df.set_index('frame_nmr').T.to_dict('list')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

license_plate_model_path = os.path.join('.', 'runs', 'segment', 'train', 'weights', 'best.pt')
vehicle_model_path = 'yolov8n.pt'  # Asegúrate de tener este archivo en el directorio correcto

# Cargar modelos
license_plate_model = YOLO(license_plate_model_path)  # Modelo para placas de licencia
vehicle_model = YOLO(vehicle_model_path)  # Modelo preentrenado de YOLO para vehículos

threshold = 0.5
frame_count = 0

while ret:

    # Detectar placas de licencia
    license_plate_results = license_plate_model(frame)[0]

    # Detectar vehículos
    vehicle_results = vehicle_model(frame)[0]

    # Marcar vehículos
    for result in vehicle_results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold and vehicle_results.names[int(class_id)].lower() in ['car', 'truck', 'bus']:
            # Dibuja un rectángulo alrededor del vehículo
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            # Poner el nombre del vehículo
            cv2.putText(frame, vehicle_results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Marcar placas de licencia
    for result in license_plate_results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, license_plate_results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

            if frame_count in license_info:
                license_data = license_info[frame_count]
                license_number_str = str(license_data[4])
                cv2.putText(frame, license_number_str, (int(x1), int(y2 + 30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    out.write(frame)
    ret, frame = cap.read()
    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()
