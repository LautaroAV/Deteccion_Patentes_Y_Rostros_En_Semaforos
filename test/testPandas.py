from ultralytics import YOLO 
import cv2
from sort import Sort
import pandas as pd


# Inicialización de rastreador y modelos
mot_tracker = Sort()
coco_model = YOLO('./models/yolov8n.pt')
detector_patentes = YOLO("./models/best.pt")

#Definiciones y variables
vehiculos = [2, 3, 5, 7]
threshold = 0.5
resultados = {}
nombre_video = "a1.mp4"
seleccionar_video = "videos/" + nombre_video

# Cargar vídeo
cap = cv2.VideoCapture(seleccionar_video)

video_sin_extension = nombre_video.split('.')[0]

csv_path = './data/{}.csv'.format(video_sin_extension)
df = pd.read_csv(csv_path, usecols=['frame_nmr', 'car_id', 'car_bbox',
                                     'license_plate_bbox', 'license_plate_bbox_score',
                                     'license_number', 'license_number_score', 'license_plate_tesseract'])

# Tu código existente
license_number_counts = df.groupby(['car_id', 'license_number']).size().reset_index(name='license_number_count')
tesseract_counts = df.groupby(['car_id', 'license_plate_tesseract']).size().reset_index(name='tesseract_count')

# Fusionar los recuentos de las placas
merged_df = pd.concat([license_number_counts[['car_id', 'license_number', 'license_number_count']], 
                       tesseract_counts[['car_id', 'license_plate_tesseract', 'tesseract_count']]
                            .rename(columns={'license_plate_tesseract': 'license_number', 'tesseract_count': 'license_number_count'})])

# Agrupar y sumar los recuentos de las placas
merged_df = merged_df.groupby(['car_id', 'license_number'], as_index=False)['license_number_count'].sum()

# Ordenar el DataFrame resultante por 'car_id'

merged_df = merged_df.sort_values(by='car_id')
# Encontrar la patente más común por vehículo
most_common_plate = merged_df.loc[merged_df.groupby('car_id')['license_number_count'].idxmax()]

# Crear el diccionario car_license_mapping
car_license_mapping = dict(zip(most_common_plate['car_id'], most_common_plate['license_number']))

# Imprimir el diccionario
