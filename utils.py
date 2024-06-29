import csv

def write_csv(results, output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score',
            'license_number', 'license_number_score', 'license_plate_tesseract', 'license_plate_tesseract_score',
            'license_plate_google', 'license_plate_google_score', 'faces', 'faces_confidence'
        ])

        for frame_nmr, cars in results.items():
            for car_id, car_data in cars.items():
                if 'car' in car_data and 'license_plate' in car_data:
                    car_bbox = '[{} {} {} {}]'.format(*car_data['car']['bbox'])
                    license_plate_bbox = '[{} {} {} {}]'.format(*car_data['license_plate']['bbox'])
                    license_plate_text = car_data['license_plate']['text']
                    license_plate_text_score = car_data['license_plate']['text_score']

                    license_plate_tesseract = car_data.get('tesseract', {}).get('text', '')
                    license_plate_tesseract_score = car_data.get('tesseract', {}).get('text_score', '')

                    license_plate_google = car_data.get('google', {}).get('text', '')
                    license_plate_google_score = car_data.get('google', {}).get('text_score', '')

                    faces_paths = []
                    faces_confidences = []
                    if 'faces' in car_data:
                        for face in car_data['faces']:
                            faces_paths.append(face['path'])
                            faces_confidences.append(face['confidence'])

                    faces_paths_str = ','.join(faces_paths)
                    faces_confidences_str = ','.join(map(str, faces_confidences))

                    writer.writerow([
                        frame_nmr, car_id, car_bbox, license_plate_bbox, car_data['license_plate']['bbox_score'],
                        license_plate_text, license_plate_text_score, license_plate_tesseract, license_plate_tesseract_score,
                        license_plate_google, license_plate_google_score, faces_paths_str, faces_confidences_str
                    ])



