import string
import easyocr
import re

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for OCR correction
dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                if 'car' in results[frame_nmr][car_id] and 'license_plate' in results[frame_nmr][car_id] and 'text' in results[frame_nmr][car_id]['license_plate']:
                    f.write('{},{},{},{},{},{},{}\n'.format(
                        frame_nmr, car_id,
                        '[{} {} {} {}]'.format(*results[frame_nmr][car_id]['car']['bbox']),
                        '[{} {} {} {}]'.format(*results[frame_nmr][car_id]['license_plate']['bbox']),
                        results[frame_nmr][car_id]['license_plate']['bbox_score'],
                        results[frame_nmr][car_id]['license_plate']['text'],
                        results[frame_nmr][car_id]['license_plate']['text_score']
                    ))

def license_complies_format(text):
    """
    Check if the license plate follows the Indian format.
    Format: [2 Letters] [2 Digits] [1-3 Letters (Optional)] [1-4 Digits]
    """
    pattern = r"^[A-Z]{2}\d{2}[A-Z]{0,3}\d{1,4}$"
    return bool(re.match(pattern, text))

def format_license(text):
    """
    Format license plate by correcting OCR misreads.
    """
    formatted_text = "".join(dict_char_to_int.get(char, char) for char in text)
    return formatted_text

def read_license_plate(license_plate_crop):
    """
    Read the license plate text and format it.
    """
    detections = reader.readtext(license_plate_crop)
    
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')

        formatted_text = format_license(text)
        if license_complies_format(formatted_text):
            return formatted_text, score
    
    return None, None

def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on license plate position.
    """
    x1, y1, x2, y2, score, class_id = license_plate
    
    for xcar1, ycar1, xcar2, ycar2, car_id in vehicle_track_ids:
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return xcar1, ycar1, xcar2, ycar2, car_id
    
    return -1, -1, -1, -1, -1
