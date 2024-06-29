import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracker import Sort
from utils import write_csv
from video_processor import process_video_frames, generate_output_video
from ocr_detection import initialize_ocr_models
from face_detection import initialize_face_detector

# Inicialización de rastreador y modelos
mot_tracker = Sort()
coco_model, detector_patentes = initialize_ocr_models()
detector = initialize_face_detector()

# Definiciones y variables
VIDEO_NAME = "test.mp4"
VIDEO_PATH = f"videos/{VIDEO_NAME}"
ROSTROS_PATH = "images/rostros"
os.makedirs(ROSTROS_PATH, exist_ok=True)

# Procesar vídeo
resultados = process_video_frames(
    VIDEO_PATH, mot_tracker, coco_model, detector_patentes, detector, ROSTROS_PATH
)

# Guardar resultados en CSV
output_csv_path = f'./data/{VIDEO_NAME.split(".")[0]}.csv'
write_csv(resultados, output_csv_path)

# Procesar vídeo de salida
generate_output_video(VIDEO_PATH, output_csv_path, resultados)
