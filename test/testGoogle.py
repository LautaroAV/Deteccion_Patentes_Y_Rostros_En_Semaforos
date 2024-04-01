from google.cloud import vision
import io
import os

def detect_text(path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'D:\Lautii\API\master-reactor-419010-a70901d2dc50.json'

    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        new_text = ''.join(texts[0].description.split())

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

if __name__ == '__main__':
    path = './images/xd.png'
    detect_text(path)
