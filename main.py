import os
from tracker import Sort
from utils import write_csv
from video_processor import procesar_frames_video, generar_video_salida, procesar_frames_camara
from ocr_detection import initialize_ocr_models
from face_detection import initialize_face_detector

def main():
    # Inicialización de rastreador y modelos
    mot_tracker = Sort()
    coco_model, detector_patentes = initialize_ocr_models()
    detector = initialize_face_detector()

    # Definiciones y variables
    VIDEO_NAME = "test.mp4"
    VIDEO_PATH = f"videos/{VIDEO_NAME}"
    ROSTROS_PATH = "images/rostros"
    os.makedirs(ROSTROS_PATH, exist_ok=True)

    choice = input("Escribe 'video' para procesar un vídeo o 'webcam' para usar la cámara: ").strip().lower()
    if choice == 'webcam':
        procesar_frames_camara(mot_tracker, coco_model, detector_patentes, detector, ROSTROS_PATH)
    elif choice == 'video':
        resultados = procesar_frames_video(
            VIDEO_PATH, mot_tracker, coco_model, detector_patentes, detector, ROSTROS_PATH
        )

        output_csv_path = f'./data/{VIDEO_NAME.split(".")[0]}.csv'
        write_csv(resultados, output_csv_path)

        generar_video_salida(VIDEO_PATH, output_csv_path, resultados)
    else:
        print("Opción no válida. Por favor, escribe 'video' o 'webcam'.")

if __name__ == "__main__":
    main()