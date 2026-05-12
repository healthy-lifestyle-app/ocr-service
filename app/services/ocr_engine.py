import re
import time
from pathlib import Path
from typing import Any

import cv2
import easyocr
import numpy as np


MODEL_DIR = Path(__file__).resolve().parents[2] / '.easyocr'
USER_NETWORK_DIR = MODEL_DIR / 'user_network'

reader = easyocr.Reader(
    ['tr', 'en'],
    gpu=False,
    model_storage_directory=str(MODEL_DIR),
    user_network_directory=str(USER_NETWORK_DIR),
)

OCR_KEYWORDS = [
    'energy',
    'enerji',
    'kcal',
    'kj',
    'calorie',
    'kalori',
    'protein',
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
    'hazelnut',
    'findik',
    'fındık',
]

NUTRITION_ALLOWLIST = (
    '0123456789.,/%kjKJkcalKCALgGrRmlML '
    'enerjiENERJIyağYAGdoymuşDOYMUS'
    'karbonhidratKARBONHIDRATşekerSEKER'
    'proteinPROTEINlifLIFtuzTUZsaltSALT'
    'fatFATcarbsCARBSsugarSUGARfiberFIBER'
)

OCR_MODES = [
    (
        'general',
        {
            'decoder': 'beamsearch',
            'beamWidth': 10,
            'contrast_ths': 0.05,
            'adjust_contrast': 0.7,
            'text_threshold': 0.4,
            'low_text': 0.25,
            'link_threshold': 0.25,
            'width_ths': 0.3,
            'add_margin': 0.05,
            'mag_ratio': 1.5,
        },
    ),
    (
        'nutrition_numeric',
        {
            'decoder': 'beamsearch',
            'beamWidth': 10,
            'allowlist': NUTRITION_ALLOWLIST,
            'contrast_ths': 0.05,
            'adjust_contrast': 0.7,
            'text_threshold': 0.35,
            'low_text': 0.2,
            'link_threshold': 0.2,
            'width_ths': 0.2,
            'add_margin': 0.08,
            'mag_ratio': 2.0,
        },
    ),
]

FAST_OCR_OPTIONS = {
    'decoder': 'beamsearch',
    'beamWidth': 5,
    'contrast_ths': 0.05,
    'adjust_contrast': 0.7,
    'text_threshold': 0.35,
    'low_text': 0.25,
    'link_threshold': 0.25,
    'width_ths': 0.35,
    'add_margin': 0.04,
    'mag_ratio': 1.2,
}


def decode_image(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError('Görüntü okunamadı')

    return image


def resize_to_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest_side = max(height, width)

    if longest_side <= max_side:
        return image

    scale = max_side / longest_side
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))

    return cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )


def prepare_fast_image(image: np.ndarray) -> np.ndarray:
    resized = resize_to_max_side(image, 1280)
    return increase_contrast(resized)


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

    return cv2.cvtColor(
        cv2.merge((enhanced_l, a_channel, b_channel)),
        cv2.COLOR_LAB2BGR,
    )


def sharpen_image(image: np.ndarray) -> np.ndarray:
    kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ]
    )

    return cv2.filter2D(image, -1, kernel)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC,
    )
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)

    return cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )


def denoise_threshold_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC,
    )
    denoised = cv2.fastNlMeansDenoising(
        resized,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    return cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )


def create_preprocessing_variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    resized = resize_image(image)
    gray_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    sharpened = sharpen_image(gray_resized)
    thresholded = preprocess_image(image)
    denoise_thresholded = denoise_threshold_image(image)

    return [
        ('original', image),
        ('gray_resize_2x', gray_resized),
        ('gray_resize_2x_sharpen', sharpened),
        ('adaptive_threshold', thresholded),
        ('denoise_threshold', denoise_thresholded),
    ]


def get_bbox_sort_key(item: Any) -> tuple[float, float]:
    bbox = item[0] if isinstance(item, (list, tuple)) and item else []

    if not isinstance(bbox, (list, tuple)) or not bbox:
        return (0, 0)

    xs = []
    ys = []

    for point in bbox:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            xs.append(float(point[0]))
            ys.append(float(point[1]))

    if not xs or not ys:
        return (0, 0)

    return (min(ys), min(xs))


def normalize_ocr_item(item: Any, variant: str, mode: str) -> dict[str, Any] | None:
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return None

    bbox = item[0]
    text = item[1]
    confidence = item[2] if len(item) > 2 else 0

    if not isinstance(text, str) or not text.strip():
        return None

    if not isinstance(bbox, (list, tuple)) or not bbox:
        return None

    points = []

    for point in bbox:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            points.append([float(point[0]), float(point[1])])

    if not points:
        return None

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        conf = 0.0

    return {
        'text': text.strip(),
        'conf': conf,
        'bbox': points,
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'x_center': (x_min + x_max) / 2,
        'y_center': (y_min + y_max) / 2,
        'width': x_max - x_min,
        'height': y_max - y_min,
        'variant': variant,
        'mode': mode,
    }


def boxes_to_lines(boxes: list[dict[str, Any]]) -> list[str]:
    if not boxes:
        return []

    heights = sorted(box['height'] for box in boxes if box.get('height', 0) > 0)
    median_height = heights[len(heights) // 2] if heights else 16
    tolerance = max(10, median_height * 0.6)
    rows: list[list[dict[str, Any]]] = []

    for box in sorted(boxes, key=lambda item: item['y_center']):
        target_row = None

        for row in rows:
            row_y = sum(item['y_center'] for item in row) / len(row)

            if abs(box['y_center'] - row_y) <= tolerance:
                target_row = row
                break

        if target_row is None:
            rows.append([box])
        else:
            target_row.append(box)

    lines = []

    for row in rows:
        text = ' '.join(box['text'] for box in sorted(row, key=lambda item: item['x_min']))

        if text.strip():
            lines.append(text.strip())

    return lines


def run_easyocr(
    image: np.ndarray,
    variant: str,
    mode: str,
    options: dict[str, Any],
) -> tuple[list[str], list[float], list[dict[str, Any]]]:
    image = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    results = reader.readtext(
        image,
        detail=1,
        paragraph=False,
        **options,
    )
    boxes: list[dict[str, Any]] = []

    for item in sorted(results, key=get_bbox_sort_key):
        box = normalize_ocr_item(item, variant, mode)

        if box is not None:
            boxes.append(box)

    lines = boxes_to_lines(boxes)
    confidences = [box['conf'] for box in boxes]

    return lines, confidences, boxes


def score_ocr_result(lines: list[str], confidences: list[float]) -> float:
    if not lines:
        return 0.0

    text = '\n'.join(lines).lower()
    keyword_score = sum(1 for keyword in OCR_KEYWORDS if keyword in text)
    number_matches = re.findall(r'\d+[.,]?\d*', text)
    unit_matches = re.findall(r'\d+[.,]?\d*\s*(g|gr|mg|kcal|kj)', text)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    garbage_penalty = sum(
        1
        for line in lines
        if len(line.strip()) <= 1 or re.fullmatch(r'[^a-zA-Z0-9]+', line.strip())
    )

    return (
        keyword_score * 5
        + min(len(number_matches), 35) * 1.4
        + min(len(unit_matches), 20) * 2
        + min(len(lines), 70) * 0.7
        + avg_confidence * 15
        + min(len(text), 1800) / 1800 * 10
        - garbage_penalty * 2.5
    )


def select_best_ocr_result(
    variants: list[tuple[str, np.ndarray]],
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []

    for variant_name, variant_image in variants:
        for mode_name, options in OCR_MODES:
            try:
                lines, confidences, boxes = run_easyocr(
                    variant_image,
                    variant_name,
                    mode_name,
                    options,
                )
            except ValueError:
                continue

            score = score_ocr_result(lines, confidences)

            candidates.append(
                {
                    'variant': variant_name,
                    'mode': mode_name,
                    'score': round(score, 2),
                    'avg_confidence': round(
                        sum(confidences) / len(confidences) if confidences else 0,
                        4,
                    ),
                    'line_count': len(lines),
                    'text': '\n'.join(lines),
                    'lines': lines,
                    'boxes': boxes,
                }
            )

    candidates = sorted(candidates, key=lambda item: item['score'], reverse=True)

    return candidates[0] if candidates else {
        'variant': None,
        'score': 0,
        'avg_confidence': 0,
        'line_count': 0,
        'text': '',
        'lines': [],
        'boxes': [],
        'mode': None,
    }


def normalize_extract_mode(mode: str | None) -> str:
    return 'accurate' if mode == 'accurate' else 'fast'


def build_ocr_result(
    lines: list[str],
    confidences: list[float],
    boxes: list[dict[str, Any]],
    variant: str,
    mode: str,
) -> dict[str, Any]:
    score = score_ocr_result(lines, confidences)

    return {
        'variant': variant,
        'mode': mode,
        'score': round(score, 2),
        'avg_confidence': round(
            sum(confidences) / len(confidences) if confidences else 0,
            4,
        ),
        'line_count': len(lines),
        'text': '\n'.join(lines),
        'lines': lines,
        'boxes': boxes,
    }


def extract_fast_ocr(image: np.ndarray) -> dict[str, Any]:
    fast_image = prepare_fast_image(image)
    lines, confidences, boxes = run_easyocr(
        fast_image,
        'fast_clahe_resize',
        'fast',
        FAST_OCR_OPTIONS,
    )

    return build_ocr_result(
        lines,
        confidences,
        boxes,
        'fast_clahe_resize',
        'fast',
    )


def extract_text_from_image(image_bytes: bytes, mode: str = 'fast'):
    started_at = time.perf_counter()
    extract_mode = normalize_extract_mode(mode)
    image = decode_image(image_bytes)
    original_height, original_width = image.shape[:2]

    if extract_mode == 'accurate':
        resized_image = resize_to_max_side(image, 1600)
        best = select_best_ocr_result(create_preprocessing_variants(resized_image))
    else:
        resized_image = resize_to_max_side(image, 1280)
        best = extract_fast_ocr(image)

    resized_height, resized_width = resized_image.shape[:2]
    processing_time_ms = round((time.perf_counter() - started_at) * 1000, 2)

    print(
        '[OCR] mode=%s original=%sx%s resized=%sx%s elapsed_ms=%.2f'
        % (
            extract_mode,
            original_width,
            original_height,
            resized_width,
            resized_height,
            processing_time_ms,
        )
    )

    return {
        'text': best['text'],
        'lines': best['lines'],
        'ocr_boxes': best.get('boxes', []),
        'processing_time_ms': processing_time_ms,
        'ocr_debug': {
            'variant': best['variant'],
            'mode': best.get('mode'),
            'requested_mode': extract_mode,
            'score': best['score'],
            'line_count': best.get('line_count', 0),
            'avg_confidence': best.get('avg_confidence', 0),
            'original_size': {
                'width': original_width,
                'height': original_height,
            },
            'resized_size': {
                'width': resized_width,
                'height': resized_height,
            },
            'processing_time_ms': processing_time_ms,
        },
    }
