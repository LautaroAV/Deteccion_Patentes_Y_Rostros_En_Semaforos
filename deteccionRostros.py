import cv2
import os

# Define la ruta de la carpeta donde se guardarán las imágenes de rostros encontrados
rostros_encontrados_path = "images/rostros"

# Verifica si la carpeta existe, si no, la crea
if not os.path.exists(rostros_encontrados_path):
    print('Carpeta creada:', rostros_encontrados_path)
    os.makedirs(rostros_encontrados_path)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
count = 0

# Inicia la captura de video desde la cámara
cap = cv2.VideoCapture(1)  # 0 representa la cámara predeterminada

while True:
    ret, frame = cap.read()  # Captura un frame de la cámara

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceClassif.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (128, 0, 255), 2)
        # Guardar el rostro detectado
        rostro = frame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('{}/rostro_{}.jpg'.format(rostros_encontrados_path, count), rostro)
        count += 1

    cv2.rectangle(frame, (10, 5), (450, 25), (255, 255, 255), -1)
    cv2.putText(frame, 'Presione q para salir', (10, 20), 2, 0.5, (128, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)  # Espera 1 milisegundo por la tecla presionada

    if k == ord('q'):
        break

# Libera la captura de video y cierra todas las ventanas
cap.release()
cv2.destroyAllWindows()
