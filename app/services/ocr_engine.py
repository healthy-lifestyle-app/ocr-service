import os

os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'

import cv2
import numpy as np
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_angle_cls=True,
    lang='tr',
    det_limit_side_len=1600,
    drop_score=0.25,
)


def resize_image(image, scale=4):
    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )


def sharpen_image(image):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ])

    return cv2.filter2D(image, -1, kernel)


def crop_center_product_area(image):
    height, width = image.shape[:2]

    # Paketin alt-orta/sağ kısmına odaklanır.
    # Besin tablosu genelde burada olduğu için daha iyi okur.
    x1 = int(width * 0.02)
    x2 = int(width * 0.98)
    y1 = int(height * 0.45)
    y2 = int(height * 0.82)

    return image[y1:y2, x1:x2]


def preprocess_original(image):
    resized = resize_image(image, 2)
    return sharpen_image(resized)


def preprocess_gray_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = resize_image(gray, 2)

    blurred = cv2.GaussianBlur(resized, (3, 3), 0)

    processed = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    return processed


def preprocess_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = resize_image(gray, 2)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    contrast = clahe.apply(resized)
    sharp = sharpen_image(contrast)

    return sharp


def preprocess_denoised(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = resize_image(gray, 2)

    denoised = cv2.fastNlMeansDenoising(
        resized,
        None,
        h=12,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    sharp = sharpen_image(denoised)

    return sharp


def collect_texts_from_result(result):
    texts = []

    if not result or not isinstance(result, list):
        return texts

    for page in result:
        if not isinstance(page, list):
            continue

        for line in page:
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                continue

            text_info = line[1]

            if isinstance(text_info, (list, tuple)) and len(text_info) >= 1:
                text = text_info[0]

                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())

    return texts


def run_ocr(image):
    result = ocr.ocr(image, cls=True)
    return collect_texts_from_result(result)


def score_lines(lines):
    if not lines:
        return -999

    keywords = [
        'nutrition',
        'facts',
        'besin',
        'değeri',
        'enerji',
        'energy',
        'kalori',
        'calorie',
        'yağ',
        'fat',
        'doymuş',
        'saturated',
        'karbonhidrat',
        'carbohydrate',
        'şeker',
        'sugar',
        'lif',
        'fiber',
        'protein',
        'tuz',
        'salt',
        'sodyum',
        'sodium',
        'ingredients',
        'içindekiler',
        'alerjen',
        'allergen',
        'contains',
        'içerir',
    ]

    text = '\n'.join(lines).lower()

    keyword_score = sum(10 for keyword in keywords if keyword in text)
    length_score = min(len(text) / 20, 30)

    garbage_chars = sum(
        1
        for ch in text
        if ch in ['~', '`', '^', '|', '{', '}', '[', ']', '<', '>']
    )

    garbage_penalty = garbage_chars * 3
    very_short_penalty = sum(1 for line in lines if len(line.strip()) <= 2) * 2

    return keyword_score + length_score - garbage_penalty - very_short_penalty


def unique_lines(lines):
    result = []
    seen = set()

    for line in lines:
        cleaned = line.strip()

        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)

    return result


def extract_text_from_image(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError('Görüntü okunamadı')

    cropped = crop_center_product_area(image)

    variants = [
        image,
        cropped,
        preprocess_original(cropped),
        preprocess_gray_threshold(cropped),
        preprocess_contrast(cropped),
        preprocess_denoised(cropped),
    ]

    best_lines = []
    best_score = -999

    for variant in variants:
        try:
            lines = run_ocr(variant)
            score = score_lines(lines)

            if score > best_score:
                best_score = score
                best_lines = lines
        except Exception as e:
            print('OCR varyant hatası:', str(e))
            continue

    final_lines = unique_lines(best_lines)

    return {
        'text': '\n'.join(final_lines),
        'lines': final_lines,
        'score': best_score,
    }