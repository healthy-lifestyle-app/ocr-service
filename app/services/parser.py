import re
from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# ALLERGEN & NUTRIENT KEYWORD TABLES
# ---------------------------------------------------------------------------

ALLERGEN_KEYWORDS = {
    'GLUTEN': [
        'gluten',
        'glute',
        'gluten iceren tahillar',
        'arpa',
        'cavdar',
        'yulaf',
    ],
    'WHEAT': [
        'wheat',
        'wheat flour',
        'bugday',
        'bugday unu',
    ],
    'PEANUT': [
        'peanut',
        'peanuts',
        'peanuis',
        'yer fistigi',
        'yer fistik',
        'yerfst',
        'fistik',
    ],
    'SOY': [
        'soy',
        'soya',
        'soya lesitini',
        'soy lecithin',
        'lesitin',
        'lecithin',
    ],
    'MILK': [
        'milk',
        'milk solids',
        'sut',
        'sutlu',
        'sut urunu',
        'sut urunleri',
        'laktoz',
        'lactose',
        'whey',
        'peynir alti suyu',
        'kazein',
        'casein',
    ],
    'HAZELNUT': [
        'hazelnut',
        'hazelnuts',
        'hazehnut',
        'findik',
    ],
    'ALMOND': [
        'almond',
        'almonds',
        'almnd',
        'badem',
    ],
    'WALNUT': [
        'walnut',
        'walnuts',
        'ceviz',
    ],
    'CASHEW': [
        'cashew',
        'cashews',
        'kaju',
    ],
    'PISTACHIO': [
        'pistachio',
        'pistachios',
        'stachio',
        'antep fistigi',
    ],
    'EGG': [
        'egg',
        'eggs',
        'yumurta',
    ],
    'FISH': [
        'fish',
        'balik',
    ],
    'SESAME': [
        'sesame',
        'sesam',
        'susam',
    ],
    'MUSTARD': [
        'mustard',
        'hardal',
    ],
    'CELERY': [
        'celery',
        'kereviz',
    ],
    'SULFITES': [
        'sulfites',
        'sulphites',
        'sulfit',
    ],
    'LUPIN': [
        'lupin',
        'aci bakla',
    ],
    'CRUSTACEANS': [
        'crustaceans',
        'crustacean',
        'kabuklular',
        'kabuklu',
        'karides',
        'yengec',
    ],
    'MOLLUSCS': [
        'molluscs',
        'mollusks',
        'midye',
        'istiridye',
        'kalamar',
    ],
}

NUTRIENT_RANGES = {
    'calories':      (0, 900),
    'fat':           (0, 100),
    'saturated_fat': (0, 100),
    'carb':          (0, 100),
    'sugar':         (0, 100),
    'fiber':         (0, 100),
    'protein':       (0, 100),
    'salt':          (0, 20),
}


# ---------------------------------------------------------------------------
# GÖRÜNTÜ ÖN İŞLEME
# ---------------------------------------------------------------------------

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    EasyOCR doğruluğunu artırmak için görüntüyü ön işler.
    BGR veya RGB numpy array kabul eder.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh


def run_easyocr(reader, img: np.ndarray, confidence_threshold: float = 0.3):
    """
    EasyOCR'ı çalıştırır, confidence filtresi uygular.
    Döner: (text: str, lines: list[str])

    Kullanım:
        import easyocr, cv2
        reader = easyocr.Reader(['tr', 'en'])
        img = cv2.imread('label.jpg')
        text, lines = run_easyocr(reader, img)
        result = extract_nutrition_data(text, lines)
    """
    processed = preprocess_image(img)
    raw_result = reader.readtext(processed)

    filtered = [item for item in raw_result if item[2] >= confidence_threshold]
    if not filtered:
        filtered = [item for item in raw_result if item[2] >= 0.1]

    lines = [item[1] for item in filtered]
    text = '\n'.join(lines)

    return text, lines


# ---------------------------------------------------------------------------
# METİN NORMALİZASYONU
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Türkçe karakterleri ASCII'ye çevirir, küçük harf yapar."""
    if not text:
        return ''

    text = str(text).lower()

    replacements = {
        'ı': 'i', 'İ': 'i',
        'ğ': 'g', 'Ğ': 'g',
        'ü': 'u', 'Ü': 'u',
        'ş': 's', 'Ş': 's',
        'ö': 'o', 'Ö': 'o',
        'ç': 'c', 'Ç': 'c',
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def fix_common_ocr_errors(text: str) -> str:
    """
    normalize_text önce çalışır (Türkçe → ASCII),
    ardından bilinen OCR hatalarını düzeltir.
    Tüm key'ler normalize edilmiş (ASCII) halde.
    """
    normalized = normalize_text(text)

    replacements = {
        # Genel ayraçlar
        '|': ' ', ';': ' ', '：': ':', '–': '-', '—': '-',

        # Kalori / enerji
        'enerii': 'enerji',
        'enerjı': 'enerji',
        'enerji ve besin oan': 'enerji ve besin degeri',
        'enerive': 'enerji ve',
        'energyy': 'energy',
        'kalori': 'calories',
        'kkal': 'kcal',
        'kca l': 'kcal',
        'kca1': 'kcal',
        'kcai': 'kcal',
        'k cal': 'kcal',
        'kcal)': 'kcal',

        # Yağ
        'yao': 'yag',
        'tag': 'yag',
        'yaf': 'yag',
        'yaq': 'yag',
        'yaglar': 'yag',

        # Doymuş yağ
        'doymusyag': 'doymus yag',
        'sahated': 'saturated',
        'saturaled': 'saturated',
        'saturates': 'saturated',
        'sat fat': 'saturated fat',

        # Karbonhidrat
        'karbohidrat': 'karbonhidrat',
        'karbon hidrat': 'karbonhidrat',
        'karber': 'karbonhidrat',
        'carbohydrat': 'carbohydrate',
        'carbohydratee': 'carbohydrate',
        'carbonhydrate': 'carbohydrate',
        'carbo hydrate': 'carbohydrate',
        'sugars': 'sugar',

        # Şeker
        'seke': 'seker',
        'skar': 'seker',
        'sekerier': 'sekerler',
        'scker': 'seker',

        # Protein
        'proteın': 'protein',
        'prolein': 'protein',
        'proteln': 'protein',
        'protcin': 'protein',
        'proicn': 'protein',

        # Lif
        'lifler': 'lif',
        'libre': 'fibre',
        'fbre': 'fibre',
        'flbre': 'fibre',
        'fber': 'fiber',

        # Tuz
        'sait': 'salt',
        'sali': 'salt',
        'salz': 'salt',
        'sodlum': 'sodium',

        # İçindekiler
        'igindekiler': 'icindekiler',
        'bar-igindekiler': 'icindekiler',
        'bar-icindekiler': 'icindekiler',

        # Fıstık
        'fistiku': 'fistik',
        'fistiki': 'fistik',
        'fistigi': 'fistik',
        'fistikli': 'fistik',
        'yerfg': 'yer fistik',
        'yer fg': 'yer fistik',
        'yerfq': 'yer fistik',
        'yerfst': 'yer fistik',
        'peanuis': 'peanuts',

        # Süt / çikolata
        'stlu': 'sutlu',
        'sutiu': 'sutlu',
        'suttu': 'sutlu',
        'sudli': 'sutlu',
        'stlikolata': 'sutlu cikolata',
        'ikolata': 'cikolata',

        # Kuruyemiş
        'almndstachio': 'almond pistachio',
        'almondstachio': 'almond pistachio',
        'almondpistachio': 'almond pistachio',
        'almondpisschio': 'almond pistachio',
        'almondpisfachio': 'almond pistachio',
        'almnd': 'almond',
        'pistacio': 'pistachio',
        'pisschio': 'pistachio',
        'pissachio': 'pistachio',
        'pisfachio': 'pistachio',
        'hazalnut': 'hazelnut',
        'hasalnut': 'hazelnut',
        'hazeinut': 'hazelnut',
        'hazehnut': 'hazelnut',
        'soypeanut': 'soy peanut',
        'sesam': 'sesame',

        # Gluten
        'glufen': 'gluten',
        'guten': 'gluten',
        'giuten': 'gluten',
        'gluken': 'gluten',
        'gulen': 'gluten',
    }

    # Uzun yanlışları önce düzelt
    for wrong in sorted(replacements.keys(), key=len, reverse=True):
        normalized = normalized.replace(wrong, replacements[wrong])

    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = normalized.replace(' / ', '/')

    return normalized.strip()


# ---------------------------------------------------------------------------
# SAYI ARAÇLARI
# ---------------------------------------------------------------------------

def to_float(value) -> float | None:
    if value is None:
        return None

    value = str(value).strip()
    value = value.replace(',', '.')
    value = value.replace('O', '0').replace('o', '0')
    value = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', value)

    match = re.search(r'\d+(?:\.\d+)?', value)
    if not match:
        return None

    try:
        return float(match.group(0))
    except ValueError:
        return None


def extract_numbers(line: str) -> list[float]:
    if not line:
        return []

    matches = re.findall(r'\d+(?:[.,]\d+)?', str(line))
    numbers = []

    for match in matches:
        value = to_float(match)
        if value is not None:
            numbers.append(value)

    return numbers


def clean_number(value: Any):
    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    value = round(value, 3)

    if float(value).is_integer():
        return int(value)

    return value


def normalize_nutrient_value(value: float | None, nutrient: str):
    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    # Tuz: OCR bazen 0.43 → 43 gibi okuyabiliyor
    if nutrient == 'salt':
        if 10 < value < 100:
            value = value / 10
        elif value >= 100:
            value = value / 100

    # Protein / lif: OCR bazen 6.5 → 65 gibi okuyabiliyor
    if nutrient in {'protein', 'fiber'}:
        if 60 < value < 100:
            value = value / 10

    min_value, max_value = NUTRIENT_RANGES.get(nutrient, (0, 100))

    if value < min_value or value > max_value:
        return None

    return clean_number(value)


# ---------------------------------------------------------------------------
# SATIR ARAÇLARI
# ---------------------------------------------------------------------------

def get_lines_from_text(text: str) -> list[str]:
    if not text:
        return []

    raw_lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    return [fix_common_ocr_errors(line) for line in raw_lines if line.strip()]


# ---------------------------------------------------------------------------
# FİLTRE FONKSİYONLARI
# ---------------------------------------------------------------------------

def is_noise_line(line: str) -> bool:
    lower_line = fix_common_ocr_errors(line)

    noise_words = [
        'prod.code', 'prod code', 'cesit no', 'registration',
        'kayit', 'ref.no', 'refno', 'relno', 'barcode',
        'www', 'http', '.com', 'iletisim',
        'ureticifirma', 'uretici firma', 'san.', 'tic.',
        'mahallesi', 'caddesi', 'sokak', 'istanbul', 'turkiye',
        'origin', 'mensei', 'lot', 'tett', 'bbd', 'best before',
        'ambalaj', 'packaging', 'halal', 'certificated',
        'sertifika', 'gida kodeksi', 'turk gida', 'registration number',
    ]

    return any(word in lower_line for word in noise_words)


def is_reference_line(line: str) -> bool:
    """
    Sadece başlık/porsiyon satırlarını filtreler.
    İçinde besin kelimesi varsa filtreleme — değer içeriyor olabilir.
    """
    lower_line = fix_common_ocr_errors(line)

    nutrient_words = [
        'energy', 'enerji', 'kcal', 'kj',
        'yag', 'fat', 'doymus', 'saturated',
        'karbonhidrat', 'carbohydrate',
        'seker', 'sugar', 'lif', 'fiber', 'fibre',
        'protein', 'tuz', 'salt',
    ]

    if any(word in lower_line for word in nutrient_words):
        return False

    reference_patterns = [
        r'\b100\s*(g|gr|gram|ml)\b',
        r'\bper\s*100\s*(g|gr|gram|ml)\b',
        r'\b\d+\s*(g|gr|gram|ml)\s*%\s*ra\b',
        r'\bamount\s*per\b',
        r'\bnutrition\s*facts\b',
        r'\bbesin\s*ogeleri\b',
        r'\bnet\s*weight\b',
    ]

    return any(re.search(pattern, lower_line) for pattern in reference_patterns)


def is_invalid_candidate_line(line: str) -> bool:
    if not line or not str(line).strip():
        return True
    if is_noise_line(line):
        return True
    if is_reference_line(line):
        return True
    return False


# ---------------------------------------------------------------------------
# İÇERİK & ALERJEN
# ---------------------------------------------------------------------------

def extract_ingredient_text(text: str) -> str:
    if not text:
        return ''

    normalized = fix_common_ocr_errors(text)

    start_keywords = [
        'icindekiler', 'ingredients', 'alerjen', 'allergen',
        'contains', 'may contain', 'icerir', 'icerebilir',
    ]

    stop_keywords = [
        'nutrition facts', 'besin ogeleri', 'enerji',
        'energy', 'amount per', 'net weight',
        'tavsiye edilen', 'best before',
    ]

    best_index = None

    for keyword in start_keywords:
        index = normalized.find(fix_common_ocr_errors(keyword))
        if index != -1 and (best_index is None or index < best_index):
            best_index = index

    if best_index is None:
        return text.strip()

    ingredient_part = text[best_index:]
    normalized_ingredient_part = fix_common_ocr_errors(ingredient_part)

    stop_index = None

    for keyword in stop_keywords:
        index = normalized_ingredient_part.find(fix_common_ocr_errors(keyword))
        if index > 20 and (stop_index is None or index < stop_index):
            stop_index = index

    if stop_index is not None:
        ingredient_part = ingredient_part[:stop_index]

    return ingredient_part.strip()


def detect_allergens(text: str) -> list[str]:
    normalized_text = fix_common_ocr_errors(text)
    detected = []

    for allergen_code, keywords in ALLERGEN_KEYWORDS.items():
        for keyword in keywords:
            normalized_keyword = fix_common_ocr_errors(keyword)
            if not normalized_keyword:
                continue

            boundary_pattern = rf'(?<![a-zA-Z]){re.escape(normalized_keyword)}(?![a-zA-Z])'
            substring_allowed = len(normalized_keyword) >= 5

            if re.search(boundary_pattern, normalized_text) or (
                substring_allowed and normalized_keyword in normalized_text
            ):
                detected.append(allergen_code)
                break

    return detected


# ---------------------------------------------------------------------------
# DEĞER BULMA — ORTAK YAPI
# ---------------------------------------------------------------------------

def choose_best_number(numbers: list[float], nutrient: str) -> float | None:
    """
    Sayı listesinden 100g değerini seç.
    Porsiyon başlıklarını ve 100g başlığını atla, ilk geçerli sayıyı döndür.
    """
    if not numbers:
        return None

    cleaned = []

    for number in numbers:
        if number == 100:
            continue

        # Porsiyon gramajı olabilecek sayılar — yağ hariç atla
        if nutrient != 'fat' and number in {25, 30, 34, 36, 40, 50}:
            continue

        normalized = normalize_nutrient_value(number, nutrient)
        if normalized is not None:
            cleaned.append(normalized)

    return cleaned[0] if cleaned else None


def find_value_by_keywords(
    lines: list[str],
    keywords: list[str],
    nutrient: str,
    search_window: int = 3,
) -> float | None:
    """
    Satırları tara, keyword bulunan satırdan itibaren search_window satır içinde
    geçerli sayıyı bul.
    """
    normalized_keywords = [fix_common_ocr_errors(k) for k in keywords]

    for index, line in enumerate(lines):
        lower_line = fix_common_ocr_errors(line)

        if not any(kw in lower_line for kw in normalized_keywords):
            continue

        # Yağ satırı doymuş yağa kaymasın
        if nutrient == 'fat' and ('doymus' in lower_line or 'saturated' in lower_line):
            continue

        # Karbonhidrat satırı şekere kaymasın
        if nutrient == 'carb' and ('seker' in lower_line or 'sugar' in lower_line):
            continue

        window = lines[index: index + search_window + 1]

        for candidate_line in window:
            if is_invalid_candidate_line(candidate_line):
                continue

            candidate_lower = fix_common_ocr_errors(candidate_line)

            # Yağ penceresinde doymuş yağa geçme
            if nutrient == 'fat' and ('doymus' in candidate_lower or 'saturated' in candidate_lower):
                continue

            numbers = extract_numbers(candidate_lower)
            value = choose_best_number(numbers, nutrient)

            if value is not None:
                return value

    return None


# ---------------------------------------------------------------------------
# KALORİ — ÖZEL MANTIK
# ---------------------------------------------------------------------------

def find_calories(lines: list[str], text: str) -> int | float | None:
    clean_text = fix_common_ocr_errors(text)

    # 1. "446 kcal" gibi doğrudan eşleşme — en güvenilir yol
    kcal_matches = re.findall(
        r'(\d+(?:[.,]\d+)?)\s*(kcal|kkal)',
        clean_text,
        re.IGNORECASE,
    )

    for match in kcal_matches:
        value = to_float(match[0])
        if value is not None and 180 <= value <= 900:
            return clean_number(value)

    # 2. Enerji satırında kJ ve kcal karışıksa — kcal inline ara
    for index, line in enumerate(lines):
        lower_line = fix_common_ocr_errors(line)

        if not any(word in lower_line for word in ['enerji', 'energy', 'kcal', 'kj', 'calories']):
            continue

        # Sadece kJ satırını atla
        if 'kj' in lower_line and 'kcal' not in lower_line:
            continue

        window = lines[max(0, index - 1): min(index + 4, len(lines))]

        for current_line in window:
            if is_invalid_candidate_line(current_line):
                continue

            current_lower = fix_common_ocr_errors(current_line)

            # Satır içinde "kcal" varsa o sayıyı al
            kcal_inline = re.search(
                r'(\d+(?:[.,]\d+)?)\s*(kcal|kkal)',
                current_lower,
                re.IGNORECASE,
            )
            if kcal_inline:
                value = to_float(kcal_inline.group(1))
                if value is not None and 180 <= value <= 900:
                    return clean_number(value)

            numbers = extract_numbers(current_lower)
            # 100g değeri tabloda daima ilk gelir, porsiyon değeri sonra
            candidates = [n for n in numbers if 180 <= n <= 900]

            if candidates:
                return clean_number(candidates[0])

    # 3. Son çare: tüm metinde 180-900 arası ilk sayı
    all_numbers = extract_numbers(clean_text)
    candidates = [n for n in all_numbers if 180 <= n <= 900]

    if candidates:
        return clean_number(candidates[0])

    return None


# ---------------------------------------------------------------------------
# BESİN DEĞERİ BULUCULAR
# ---------------------------------------------------------------------------

def find_fat_value(lines: list[str]) -> float | None:
    return find_value_by_keywords(
        lines,
        ['yag/fat', 'yag', 'fat'],
        'fat',
        search_window=3,
    )


def find_saturated_fat_value(lines: list[str]) -> float | None:
    return find_value_by_keywords(
        lines,
        ['doymus yag', 'saturated fat', 'saturated', 'doymus'],
        'saturated_fat',
        search_window=4,
    )


def find_carb_value(lines: list[str]) -> float | None:
    return find_value_by_keywords(
        lines,
        ['karbonhidrat', 'carbohydrate', 'carbs', 'carb'],
        'carb',
        search_window=3,
    )


def find_sugar_value(lines: list[str]) -> float | None:
    return find_value_by_keywords(
        lines,
        ['sekerler', 'seker', 'sugar'],
        'sugar',
        search_window=4,
    )


def find_fiber_value(lines: list[str]) -> float | None:
    return find_value_by_keywords(
        lines,
        ['lif', 'fiber', 'fibre'],
        'fiber',
        search_window=3,
    )


def find_protein_value(lines: list[str]) -> float | None:
    return find_value_by_keywords(
        lines,
        ['protein'],
        'protein',
        search_window=4,
    )


def find_salt_value(lines: list[str]) -> float | None:
    return find_value_by_keywords(
        lines,
        ['tuz', 'salt', 'sodium'],
        'salt',
        search_window=3,
    )


# ---------------------------------------------------------------------------
# ANA PARSER
# ---------------------------------------------------------------------------

def extract_nutrition_data(text: str, lines: list[str] | None = None) -> dict:
    """
    Ana parser — iki kullanım şekli:

    A) Sadece metin:
        result = extract_nutrition_data(text)

    B) run_easyocr ile (önerilen):
        text, lines = run_easyocr(reader, img)
        result = extract_nutrition_data(text, lines)
    """
    raw_text = text or ''

    if lines:
        fixed_lines = [fix_common_ocr_errors(line) for line in lines if line.strip()]
        fixed_text = fix_common_ocr_errors(raw_text + '\n' + '\n'.join(lines))
    else:
        fixed_lines = get_lines_from_text(raw_text)
        fixed_text = fix_common_ocr_errors(raw_text)

    ingredient_text = extract_ingredient_text(raw_text)
    detected_allergens = detect_allergens(raw_text)

    carb = find_carb_value(fixed_lines)

    return {
        'calories':           clean_number(find_calories(fixed_lines, fixed_text)),
        'fat':                clean_number(find_fat_value(fixed_lines)),
        'saturated_fat':      clean_number(find_saturated_fat_value(fixed_lines)),
        'carb':               clean_number(carb),
        'carbs':              clean_number(carb),
        'sugar':              clean_number(find_sugar_value(fixed_lines)),
        'fiber':              clean_number(find_fiber_value(fixed_lines)),
        'protein':            clean_number(find_protein_value(fixed_lines)),
        'salt':               clean_number(find_salt_value(fixed_lines)),
        'ingredients_text':   ingredient_text,
        'detected_allergens': detected_allergens,
        'has_allergen_risk':  len(detected_allergens) > 0,
    }