import cv2
from mtcnn.mtcnn import MTCNN

def initialize_face_detector():
    return MTCNN()

def detect_faces_in_frame(frame, frame_count, resultados, detector, rostros_path):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced_frame = cv2.equalizeHist(gray)
    blurred_frame = cv2.GaussianBlur(enhanced_frame, (5, 5), 0)
    adjusted_frame = cv2.convertScaleAbs(blurred_frame, alpha=1.5, beta=50)
    rgb_frame = cv2.cvtColor(adjusted_frame, cv2.COLOR_GRAY2RGB)
    faces = [face for face in detector.detect_faces(rgb_frame) if face['confidence'] > 0.85]

    for face in faces:
        x, y, w, h = face['box']
        confidence = face['confidence']
        face_assigned = False
        for auto_id, data in resultados[frame_count].items():
            car_bbox = data['car']['bbox']
            if x > car_bbox[0] and x < car_bbox[2] and y > car_bbox[1] and y < car_bbox[3]:
                if 'faces' not in resultados[frame_count][auto_id]:
                    resultados[frame_count][auto_id]['faces'] = []

                rostro = cv2.resize(frame[y:y + h, x:x + w], (150, 150), interpolation=cv2.INTER_CUBIC)
                rostro_path = f'{rostros_path}/rostro_{frame_count}_{auto_id}.jpg'
                cv2.imwrite(rostro_path, rostro)
                resultados[frame_count][auto_id]['faces'].append({
                    'path': rostro_path,
                    'confidence': confidence
                })

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                face_assigned = True
                break
