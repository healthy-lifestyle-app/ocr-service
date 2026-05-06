import os
import re
from typing import Any

os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'

import cv2
import numpy as np
from paddleocr import PaddleOCR


ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
)


OCR_KEYWORDS = [
    'energy',
    'enerji',
    'kcal',
    'kj',
    'calorie',
    'kalori',
    'protein',
    'proteinler',
    'carbohydrate',
    'karbonhidrat',
    'carbs',
    'sugar',
    'seker',
    'şeker',
    'fat',
    'yag',
    'yağ',
    'saturated',
    'doymus',
    'doymuş',
    'fiber',
    'lif',
    'salt',
    'tuz',
    'sodium',
    'ingredients',
    'icindekiler',
    'içindekiler',
    'allergen',
    'alerjen',
    'milk',
    'sut',
    'süt',
    'gluten',
    'wheat',
    'bugday',
    'buğday',
    'soy',
    'soya',
    'peanut',
    'yer fistigi',
    'yer fıstığı',
    'almond',
    'badem',
    'pistachio',
    'antep',
    'walnut',
    'ceviz',
    'hazelnut',
    'findik',
    'fındık',
    'sesame',
    'susam',
]


def decode_image(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError('Görüntü okunamadı')

    return image


def calculate_image_quality(image: np.ndarray) -> dict[str, Any]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    overexposed_ratio = float(np.mean(gray > 245))
    underexposed_ratio = float(np.mean(gray < 25))

    warnings: list[str] = []

    if blur_score < 80:
        warnings.append('Fotoğraf bulanık görünüyor. Lütfen etiketi daha net çekin.')

    if contrast < 35:
        warnings.append('Kontrast düşük. Etiket yazıları yeterince belirgin olmayabilir.')

    if overexposed_ratio > 0.08:
        warnings.append('Fotoğrafta parlama/ışık patlaması var. Işığı azaltıp tekrar çekin.')

    if underexposed_ratio > 0.20:
        warnings.append('Fotoğraf karanlık görünüyor. Daha aydınlık ortamda tekrar çekin.')

    return {
        'blur_score': round(blur_score, 2),
        'brightness': round(brightness, 2),
        'contrast': round(contrast, 2),
        'overexposed_ratio': round(overexposed_ratio, 4),
        'underexposed_ratio': round(underexposed_ratio, 4),
        'warnings': warnings,
        'is_acceptable': len(warnings) == 0,
    }


def resize_image(image: np.ndarray, scale: float = 2.0) -> np.ndarray:
    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )


def increase_contrast(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def sharpen_image(image: np.ndarray) -> np.ndarray:
    kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ]
    )

    return cv2.filter2D(image, -1, kernel)


def denoise_image(image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h=8,
        hColor=8,
        templateWindowSize=7,
        searchWindowSize=21,
    )


def threshold_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresholded = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    return cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)


def reduce_glare(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)

    glare_mask = cv2.inRange(v_channel, 235, 255)
    glare_mask = cv2.dilate(glare_mask, np.ones((3, 3), np.uint8), iterations=1)

    repaired = cv2.inpaint(image, glare_mask, 3, cv2.INPAINT_TELEA)
    return repaired


def create_preprocessing_variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    resized = resize_image(image)

    contrast = increase_contrast(resized)
    denoised = denoise_image(resized)
    sharpened = sharpen_image(contrast)
    thresholded = threshold_image(contrast)
    glare_reduced = reduce_glare(resized)
    glare_contrast = increase_contrast(glare_reduced)

    return [
        ('original', image),
        ('resized', resized),
        ('contrast', contrast),
        ('denoise', denoised),
        ('sharpen', sharpened),
        ('threshold', thresholded),
        ('glare_reduced', glare_reduced),
        ('glare_contrast', glare_contrast),
    ]


def parse_paddle_result(result: Any) -> tuple[list[str], list[float]]:
    texts: list[str] = []
    confidences: list[float] = []

    if not result or not isinstance(result, list):
        return texts, confidences

    first_page = result[0] if len(result) > 0 else []

    if not isinstance(first_page, list):
        return texts, confidences

    for line in first_page:
        if not isinstance(line, (list, tuple)) or len(line) < 2:
            continue

        text_info = line[1]

        if not isinstance(text_info, (list, tuple)) or len(text_info) < 1:
            continue

        text = text_info[0]
        confidence = text_info[1] if len(text_info) > 1 else 0

        if isinstance(text, str) and text.strip():
            texts.append(text.strip())

            try:
                confidences.append(float(confidence))
            except (TypeError, ValueError):
                confidences.append(0.0)

    return texts, confidences


def run_ocr(image: np.ndarray) -> tuple[list[str], list[float]]:
    try:
        result = ocr.ocr(image, cls=True)
    except TypeError:
        result = ocr.ocr(image)

    return parse_paddle_result(result)


def score_ocr_result(lines: list[str], confidences: list[float]) -> float:
    if not lines:
        return 0.0

    text = '\n'.join(lines).lower()

    keyword_score = sum(1 for keyword in OCR_KEYWORDS if keyword in text)

    number_matches = re.findall(r'\d+[.,]?\d*', text)
    number_score = min(len(number_matches), 25)

    gram_matches = re.findall(r'\d+[.,]?\d*\s*(g|gr|mg|kcal|kj)', text)
    unit_score = min(len(gram_matches), 20)

    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    confidence_score = avg_confidence * 20

    line_score = min(len(lines), 40) * 0.5

    text_length_score = min(len(text), 1200) / 1200 * 10

    return (
        keyword_score * 4
        + number_score * 1.5
        + unit_score * 2
        + confidence_score
        + line_score
        + text_length_score
    )


def select_best_ocr_result(
    variants: list[tuple[str, np.ndarray]],
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []

    for variant_name, variant_image in variants:
        lines, confidences = run_ocr(variant_image)
        score = score_ocr_result(lines, confidences)

        avg_confidence = (
            sum(confidences) / len(confidences)
            if confidences
            else 0.0
        )

        candidates.append(
            {
                'variant': variant_name,
                'score': round(score, 2),
                'avg_confidence': round(avg_confidence, 4),
                'line_count': len(lines),
                'text': '\n'.join(lines),
                'lines': lines,
            }
        )

    candidates = sorted(candidates, key=lambda item: item['score'], reverse=True)

    best = candidates[0] if candidates else {
        'variant': None,
        'score': 0,
        'avg_confidence': 0,
        'line_count': 0,
        'text': '',
        'lines': [],
    }

    return {
        'best': best,
        'candidates': candidates,
    }


def extract_text_from_image(image_bytes: bytes):
    image = decode_image(image_bytes)

    quality = calculate_image_quality(image)
    variants = create_preprocessing_variants(image)
    ocr_selection = select_best_ocr_result(variants)

    best = ocr_selection['best']

    return {
        'text': best['text'],
        'lines': best['lines'],
        'selected_variant': best['variant'],
        'ocr_score': best['score'],
        'avg_confidence': best['avg_confidence'],
        'image_quality': quality,
        'ocr_candidates': [
            {
                'variant': candidate['variant'],
                'score': candidate['score'],
                'avg_confidence': candidate['avg_confidence'],
                'line_count': candidate['line_count'],
            }
            for candidate in ocr_selection['candidates']
        ],
    }