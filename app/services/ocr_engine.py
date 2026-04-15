import os

os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'

import cv2
import numpy as np
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_angle_cls=False,
    lang='en',
)

def extract_text_from_image(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError('Görüntü okunamadı')

    result = ocr.ocr(image)

    texts = []

    if result and isinstance(result, list) and len(result) > 0:
        first_page = result[0]

        if isinstance(first_page, list):
            for line in first_page:
                if not isinstance(line, (list, tuple)) or len(line) < 2:
                    continue

                text_info = line[1]

                if isinstance(text_info, (list, tuple)) and len(text_info) >= 1:
                    text = text_info[0]
                    if isinstance(text, str) and text.strip():
                        texts.append(text.strip())

    return {
        'text': '\n'.join(texts),
        'lines': texts,
    }