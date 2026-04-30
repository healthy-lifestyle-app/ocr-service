import re

ALLERGEN_KEYWORDS = {
    'GLUTEN': ['gluten', 'gluten içeren tahıllar', 'gluten iceren tahillar'],
    'WHEAT': ['wheat', 'buğday', 'bugday', 'buğday unu', 'bugday unu', 'wheat flour'],
    'PEANUT': ['peanut', 'yer fıstığı', 'yer fistigi', 'yer fistik', 'fistik', 'fıstık'],
    'SOY': ['soy', 'soya', 'soya lesitini', 'lesitin', 'lecithin'],
    'MILK': [
        'milk',
        'süt',
        'sut',
        'sütlü',
        'sutlu',
        'süt ürünü',
        'sut urunu',
        'süt ürünleri',
        'sut urunleri',
        'sutlu cikolata',
        'cikolata',
        'laktoz',
        'lactose',
        'whey',
        'kazein',
    ],
    'HAZELNUT': ['hazelnut', 'fındık', 'findik'],
    'ALMOND': ['almond', 'badem'],
    'WALNUT': ['walnut', 'ceviz'],
    'CASHEW': ['cashew', 'kaju'],
    'PISTACHIO': ['pistachio', 'antep fıstığı', 'antep fistigi'],
    'EGG': ['egg', 'yumurta'],
    'FISH': ['fish', 'balık', 'balik'],
    'SESAME': ['sesame', 'susam'],
    'MUSTARD': ['mustard', 'hardal'],
    'CELERY': ['celery', 'kereviz'],
    'SULFITES': ['sulfites', 'sulphites', 'sülfit', 'sulfit'],
    'LUPIN': ['lupin', 'acı bakla', 'aci bakla'],
    'CRUSTACEANS': ['crustaceans', 'kabuklular', 'karides', 'yengeç', 'yengec'],
    'MOLLUSCS': ['molluscs', 'mollusks', 'midye', 'istiridye', 'kalamar'],
}


def normalize_text(text: str):
    text = text.lower()

    replacements = {
        'ı': 'i',
        'ğ': 'g',
        'ü': 'u',
        'ş': 's',
        'ö': 'o',
        'ç': 'c',
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def fix_common_ocr_errors(text: str):
    normalized = normalize_text(text)

    replacements = {
        'fistiku': 'fistik',
        'fistiki': 'fistik',
        'fistigi': 'fistik',
        'fistikli': 'fistik',
        'yerfg': 'yer fistik',
        'yer fg': 'yer fistik',
        'yerfq': 'yer fistik',

        'stlu': 'sutlu',
        'sutiu': 'sutlu',
        'suttu': 'sutlu',
        'sudli': 'sutlu',
        'nicerebilirstlikolata': 'icerebilir sutlu cikolata',
        'nicerebilir stlikolata': 'icerebilir sutlu cikolata',
        'stlikolata': 'sutlu cikolata',
        'ikolata': 'cikolata',
        'sutlu cikolata': 'sut',

        'susam icerebilir': 'susam icerebilir',
        'sesam': 'sesame',

        'glufen': 'gluten',
        'guten': 'gluten',
        'giuten': 'gluten',
        'gluken': 'gluten',
        'gulen': 'gluten',

        'karbohidrat': 'karbonhidrat',
        'karber': 'karbonhidrat',
        'enerii': 'enerji',
        'enerive': 'enerji ve',
        'enerii ve besin oan': 'enerji ve besin degeri',
        'besin oan': 'besin degeri',
        'yao': 'yag',
        'tag': 'yag',
        'fal': 'fat',
        'sahated': 'saturated',
        'seke': 'seker',
        'skar': 'seker',
    }

    for wrong, correct in replacements.items():
        normalized = normalized.replace(wrong, correct)

    return normalized


def to_float(value: str):
    value = value.replace(',', '.')
    try:
        return float(value)
    except ValueError:
        return None


def extract_numbers(line: str):
    matches = re.findall(r'\d+(?:[.,]\d+)?', line)
    numbers = []

    for match in matches:
        value = to_float(match)
        if value is not None:
            numbers.append(value)

    return numbers


def extract_ingredient_text(text: str):
    normalized = fix_common_ocr_errors(text)

    start_keywords = [
        'icindekiler',
        'bar-igindekiler',
        'bar-icindekiler',
        'igindekiler',
        'ingredients',
        'alerjen',
        'allergen',
        'contains',
        'may contain',
        'icerir',
        'icerebilir',
    ]

    best_index = None

    for keyword in start_keywords:
        index = normalized.find(normalize_text(keyword))

        if index != -1 and (best_index is None or index < best_index):
            best_index = index

    if best_index is not None:
        return text[best_index:]

    return text


def detect_allergens(text: str):
    normalized_text = fix_common_ocr_errors(text)
    detected = []

    for allergen_code, keywords in ALLERGEN_KEYWORDS.items():
        for keyword in keywords:
            normalized_keyword = normalize_text(keyword)
            pattern = rf'(?<![a-zA-Z]){re.escape(normalized_keyword)}(?![a-zA-Z])'

            if re.search(pattern, normalized_text):
                detected.append(allergen_code)
                break

    return detected


def is_noise_line(line: str):
    lower_line = fix_common_ocr_errors(line)

    noise_words = [
        'prod.code',
        'prod code',
        'cesit no',
        'registration',
        'kayit',
        'ref.no',
        'relno',
        'barcode',
        'www',
        'http',
        'tel',
        'hot.com',
        'taco',
        '8400',
        '2000',
        '130-137',
        'partiya',
        'parti',
        'nomrasi',
        'numarasi',
        '18-22',
        'yararlilig',
        'muddeti',
    ]

    return any(word in lower_line for word in noise_words)


def find_value_on_same_line(
    lines: list[str],
    keywords: list[str],
    max_valid_value: float | None = None,
):
    normalized_keywords = [normalize_text(keyword) for keyword in keywords]

    for line in lines:
        lower_line = fix_common_ocr_errors(line)

        if not any(keyword in lower_line for keyword in normalized_keywords):
            continue

        if is_noise_line(line):
            continue

        if '%' in line:
            continue

        if re.search(r'100\s*(g|ml|da)', lower_line):
            continue

        numbers = extract_numbers(line)

        if not numbers:
            return None

        valid_numbers = []

        for number in numbers:
            if number < 0:
                continue

            if number == 100:
                continue

            if max_valid_value is not None and number > max_valid_value:
                continue

            valid_numbers.append(number)

        if not valid_numbers:
            return None

        return valid_numbers[-1]

    return None


def find_calories(lines: list[str], text: str):
    clean_text = fix_common_ocr_errors(text)

    kcal_match = re.search(
        r'(\d+(?:[.,]\d+)?)\s*(kcal|kkal)',
        clean_text,
        re.IGNORECASE,
    )

    if kcal_match:
        value = to_float(kcal_match.group(1))
        if value is not None and 180 <= value <= 900 and value != 100:
            return value

    for i, line in enumerate(lines):
        lower_line = fix_common_ocr_errors(line)

        if not any(word in lower_line for word in ['enerji', 'energy', 'kcal', 'kkal']):
            continue

        search_area = lines[max(0, i - 2):min(i + 3, len(lines))]

        for current_line in search_area:
            current_lower = fix_common_ocr_errors(current_line)

            if is_noise_line(current_line):
                continue

            if '%' in current_line:
                continue

            if re.search(r'100\s*(g|ml|da)', current_lower):
                continue

            numbers = extract_numbers(current_line)

            candidates = [
                number
                for number in numbers
                if 180 <= number <= 900 and number != 100
            ]

            if candidates:
                return candidates[-1]

    return None


def find_fat_value(lines: list[str]):
    for line in lines:
        lower_line = fix_common_ocr_errors(line)

        if not any(keyword in lower_line for keyword in ['yag', 'fat']):
            continue

        if any(word in lower_line for word in ['trans', 'saturated', 'doymus']):
            continue

        if is_noise_line(line):
            continue

        if '%' in line:
            continue

        numbers = extract_numbers(line)

        if not numbers:
            return None

        valid_numbers = [number for number in numbers if 0 <= number <= 100 and number != 100]

        if not valid_numbers:
            return None

        return valid_numbers[-1]

    return None


def find_salt_value(lines: list[str]):
    for line in lines:
        lower_line = fix_common_ocr_errors(line)

        if not any(keyword in lower_line for keyword in ['tuz', 'salt', 'salz']):
            continue

        if is_noise_line(line):
            continue

        if '%' in line:
            continue

        numbers = extract_numbers(line)

        if not numbers:
            return None

        valid_numbers = [number for number in numbers if 0 <= number <= 10]

        if not valid_numbers:
            return None

        return valid_numbers[-1]

    return None


def extract_nutrition_data(text: str):
    fixed_text = fix_common_ocr_errors(text)
    lines = [line.strip() for line in fixed_text.split('\n') if line.strip()]

    ingredient_text = extract_ingredient_text(text)
    detected_allergens = detect_allergens(ingredient_text)

    data = {
        'calories': find_calories(lines, fixed_text),
        'protein': find_value_on_same_line(lines, ['protein'], max_valid_value=100),
        'carb': find_value_on_same_line(
            lines,
            ['karbonhidrat', 'carbohydrate', 'carb'],
            max_valid_value=100,
        ),
        'sugar': find_value_on_same_line(
            lines,
            ['seker', 'sugar', 'sugars'],
            max_valid_value=100,
        ),
        'fat': find_fat_value(lines),
        'saturated_fat': find_value_on_same_line(
            lines,
            ['doymus yag', 'saturated fat', 'saturates'],
            max_valid_value=50,
        ),
        'fiber': find_value_on_same_line(
            lines,
            ['lif', 'fiber', 'fibre'],
            max_valid_value=100,
        ),
        'salt': find_salt_value(lines),
        'ingredients_text': ingredient_text,
        'detected_allergens': detected_allergens,
        'has_allergen_risk': len(detected_allergens) > 0,
    }

    return data