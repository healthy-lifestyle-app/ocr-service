import re


ALLERGEN_KEYWORDS = {
    'GLUTEN': [
        'gluten',
        'gluten içeren tahıllar',
        'gluten iceren tahillar',
    ],
    'WHEAT': [
        'wheat',
        'wheat flour',
        'buğday',
        'bugday',
        'buğday unu',
        'bugday unu',
    ],
    'PEANUT': [
        'peanut',
        'peanuts',
        'yer fıstığı',
        'yer fistigi',
        'yer fistik',
        'fistik',
        'fıstık',
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
        'süt',
        'sut',
        'sütlü',
        'sutlu',
        'süt ürünü',
        'sut urunu',
        'süt ürünleri',
        'sut urunleri',
        'laktoz',
        'lactose',
        'whey',
        'kazein',
        'casein',
    ],
    'HAZELNUT': [
        'hazelnut',
        'fındık',
        'findik',
    ],
    'ALMOND': [
        'almond',
        'almnd',
        'badem',
    ],
    'WALNUT': [
        'walnut',
        'ceviz',
    ],
    'CASHEW': [
        'cashew',
        'kaju',
    ],
    'PISTACHIO': [
        'pistachio',
        'stachio',
        'antep fıstığı',
        'antep fistigi',
    ],
    'EGG': [
        'egg',
        'yumurta',
    ],
    'FISH': [
        'fish',
        'balık',
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
        'sülfit',
        'sulfit',
    ],
    'LUPIN': [
        'lupin',
        'acı bakla',
        'aci bakla',
    ],
    'CRUSTACEANS': [
        'crustaceans',
        'kabuklular',
        'karides',
        'yengeç',
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


def normalize_text(text: str) -> str:
    if not text:
        return ''

    text = text.lower()

    replacements = {
        'ı': 'i',
        'ğ': 'g',
        'ü': 'u',
        'ş': 's',
        'ö': 'o',
        'ç': 'c',
        'İ': 'i',
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def fix_common_ocr_errors(text: str) -> str:
    normalized = normalize_text(text)

    replacements = {
        # Peanut / pistachio / nut OCR issues
        'fistiku': 'fistik',
        'fistiki': 'fistik',
        'fistigi': 'fistik',
        'fistikli': 'fistik',
        'yerfg': 'yer fistik',
        'yer fg': 'yer fistik',
        'yerfq': 'yer fistik',

        # Milk / chocolate OCR issues
        'stlu': 'sutlu',
        'sutiu': 'sutlu',
        'suttu': 'sutlu',
        'sudli': 'sutlu',
        'stlikolata': 'sutlu cikolata',
        'ikolata': 'cikolata',

        # English nut OCR issues
        'almnd': 'almond',
        'almndstachio': 'almond pistachio',
        'almondstachio': 'almond pistachio',
        'almondpistachio': 'almond pistachio',
        'almondpisschio': 'almond pistachio',
        'almondpisfachio': 'almond pistachio',
        'pistacio': 'pistachio',
        'pistachıo': 'pistachio',
        'pisschio': 'pistachio',
        'pissachio': 'pistachio',
        'pisfachio': 'pistachio',

        # Hazelnut OCR issues
        'hazalnut': 'hazelnut',
        'hasalnut': 'hazelnut',
        'hazeinut': 'hazelnut',

        # Soy OCR issues
        'soypeanut': 'soy peanut',

        # Sesame OCR issues
        'sesam': 'sesame',

        # Gluten OCR issues
        'glufen': 'gluten',
        'guten': 'gluten',
        'giuten': 'gluten',
        'gluken': 'gluten',
        'gulen': 'gluten',

        # Nutrition OCR issues
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
        'kalori': 'calories',
        'kcal)': 'kcal',
        'kkal)': 'kcal',
        'proteın': 'protein',
        'prolein': 'protein',
        'proteln': 'protein',
        'libre': 'fibre',
        'fbre': 'fibre',
        'flbre': 'fibre',

        # Turkish dotted/undotted OCR style
        'doymus yag': 'doymus yag',
        'doymuş yag': 'doymus yag',
        'doymus yağ': 'doymus yag',
    }

    for wrong, correct in replacements.items():
        normalized = normalized.replace(wrong, correct)

    return normalized


def to_float(value: str | float | int | None):
    if value is None:
        return None

    value = str(value).strip()
    value = value.replace(',', '.')

    match = re.search(r'\d+(?:\.\d+)?', value)

    if not match:
        return None

    try:
        return float(match.group(0))
    except ValueError:
        return None


def extract_numbers(line: str) -> list[float]:
    matches = re.findall(r'\d+(?:[.,]\d+)?', line)
    numbers = []

    for match in matches:
        value = to_float(match)

        if value is not None:
            numbers.append(value)

    return numbers


def infer_decimal_from_ocr_integer(
    value: float,
    nutrient: str,
) -> float:
    """
    OCR bazen 7,2 -> 72 veya 2,3 -> 23 gibi okuyabiliyor.
    Protein/lif/tuz gibi alanlarda güvenli ondalık düzeltmesi yapar.
    Yağ/şeker/doymuş yağ için otomatik bölme yapmaz; çünkü 26, 38, 16
    gibi değerler gerçek 100 g değerleri olabilir.
    """

    if value is None:
        return value

    if not float(value).is_integer():
        return value

    value = float(value)

    if nutrient == 'salt':
        if 10 < value < 100:
            return round(value / 100, 2)

    if nutrient in {'protein', 'fiber'}:
        if 10 < value < 100:
            return round(value / 10, 2)

    return value


def normalize_nutrient_value(
    value: float,
    nutrient: str,
) -> float | None:
    if value is None:
        return None

    value = float(value)
    value = infer_decimal_from_ocr_integer(value, nutrient)

    realistic_ranges = {
        'fat': (0, 80),
        'saturated_fat': (0, 60),
        'carb': (0, 100),
        'sugar': (0, 100),
        'fiber': (0, 40),
        'protein': (0, 60),
        'salt': (0, 10),
    }

    min_value, max_value = realistic_ranges.get(nutrient, (0, 100))

    if value < min_value or value > max_value:
        return None

    return value


def is_noise_line(line: str) -> bool:
    lower_line = fix_common_ocr_errors(line)

    noise_words = [
        'prod.code',
        'prod code',
        'cesit no',
        'registration',
        'kayit',
        'ref.no',
        'refno',
        'relno',
        'barcode',
        'www',
        'http',
        '.com',
        'tel',
        'iletisim',
        'ureticifirma',
        'uretici firma',
        'firma',
        'san.',
        'tic.',
        'mahallesi',
        'caddesi',
        'sokak',
        'istanbul',
        'turkiye',
        'origin',
        'mensei',
        'lot',
        'tett',
        'bbd',
        'best before',
        'date',
        'ambalaj',
        'packaging',
        'halal',
        'certificated',
        'sertifika',
        'sertificated',
        'gida kodeksi',
        'turk gida',
        'registration number',
    ]

    return any(word in lower_line for word in noise_words)


def is_serving_or_reference_line(line: str) -> bool:
    lower_line = fix_common_ocr_errors(line)

    if re.search(r'\b100\s*(g|gr|ml|da)\b', lower_line):
        return True

    if re.search(r'\b40\s*(g|gr|ml)\b', lower_line):
        return True

    if 'amount per' in lower_line:
        return True

    if 'net weight' in lower_line:
        return True

    if '1.41oz' in lower_line:
        return True

    return False


def is_invalid_candidate_line(line: str) -> bool:
    if not line:
        return True

    if is_noise_line(line):
        return True

    if is_serving_or_reference_line(line):
        return True

    if '%' in line:
        return True

    return False


def extract_ingredient_text(text: str) -> str:
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
        'also handles products containing',
        'icerir',
        'icerebilir',
    ]

    stop_keywords = [
        'nutrition facts',
        'besin ogeleri',
        'enerji',
        'energy',
        'amount per',
        'net weight',
        'tavsiye edilen',
        'best before',
    ]

    best_index = None

    for keyword in start_keywords:
        index = normalized.find(normalize_text(keyword))

        if index != -1 and (best_index is None or index < best_index):
            best_index = index

    if best_index is None:
        return text

    ingredient_part = text[best_index:]
    normalized_ingredient_part = fix_common_ocr_errors(ingredient_part)

    stop_index = None

    for keyword in stop_keywords:
        index = normalized_ingredient_part.find(normalize_text(keyword))

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

            boundary_pattern = (
                rf'(?<![a-zA-Z]){re.escape(normalized_keyword)}(?![a-zA-Z])'
            )

            substring_allowed = len(normalized_keyword) >= 5

            if re.search(boundary_pattern, normalized_text) or (
                substring_allowed and normalized_keyword in normalized_text
            ):
                detected.append(allergen_code)
                break

    return detected


def is_value_valid_for_nutrient(
    value: float,
    min_value: float,
    max_value: float,
    allow_zero: bool = True,
) -> bool:
    if value is None:
        return False

    if not allow_zero and value == 0:
        return False

    if value < min_value:
        return False

    if value > max_value:
        return False

    if value == 100:
        return False

    return True


def find_value_on_same_line(
    lines: list[str],
    keywords: list[str],
    max_valid_value: float | None = None,
):
    normalized_keywords = [fix_common_ocr_errors(keyword) for keyword in keywords]

    for line in lines:
        lower_line = fix_common_ocr_errors(line)

        if not any(keyword in lower_line for keyword in normalized_keywords):
            continue

        if is_invalid_candidate_line(line):
            continue

        numbers = extract_numbers(line)

        if not numbers:
            continue

        valid_numbers = []

        for number in numbers:
            if number < 0:
                continue

            if number == 100:
                continue

            if max_valid_value is not None and number > max_valid_value:
                continue

            valid_numbers.append(number)

        if valid_numbers:
            return valid_numbers[-1]

    return None


def find_first_valid_number_near_label(
    lines: list[str],
    label_keywords: list[str],
    min_value: float = 0,
    max_value: float = 100,
    search_window: int = 5,
    allow_zero: bool = True,
    nutrient: str = '',
):
    normalized_keywords = [fix_common_ocr_errors(keyword) for keyword in label_keywords]

    for i, line in enumerate(lines):
        lower_line = fix_common_ocr_errors(line)

        if not any(keyword in lower_line for keyword in normalized_keywords):
            continue

        search_area = lines[i : min(i + search_window + 1, len(lines))]
        candidate_values: list[float] = []

        for candidate in search_area:
            candidate_lower = fix_common_ocr_errors(candidate)

            if is_invalid_candidate_line(candidate):
                continue

            numbers = extract_numbers(candidate_lower)

            for number in numbers:
                normalized_number = normalize_nutrient_value(number, nutrient)

                if normalized_number is None:
                    continue

                if is_value_valid_for_nutrient(
                    normalized_number,
                    min_value=min_value,
                    max_value=max_value,
                    allow_zero=allow_zero,
                ):
                    candidate_values.append(normalized_number)

        if candidate_values:
            return candidate_values[0]

    return None


def find_larger_value_near_label(
    lines: list[str],
    label_keywords: list[str],
    min_value: float = 5,
    max_value: float = 100,
    search_window: int = 6,
    exclude_keywords: list[str] | None = None,
):
    normalized_keywords = [fix_common_ocr_errors(keyword) for keyword in label_keywords]
    normalized_excludes = [
        fix_common_ocr_errors(keyword)
        for keyword in (exclude_keywords or [])
    ]

    for i, line in enumerate(lines):
        lower_line = fix_common_ocr_errors(line)

        if not any(keyword in lower_line for keyword in normalized_keywords):
            continue

        if any(keyword in lower_line for keyword in normalized_excludes):
            continue

        search_area = lines[i : min(i + search_window + 1, len(lines))]
        candidates: list[float] = []

        for candidate in search_area:
            candidate_lower = fix_common_ocr_errors(candidate)

            if is_invalid_candidate_line(candidate):
                continue

            if any(keyword in candidate_lower for keyword in normalized_excludes):
                continue

            numbers = extract_numbers(candidate_lower)

            for number in numbers:
                if min_value <= number <= max_value and number not in [40, 100]:
                    candidates.append(number)

        if candidates:
            return candidates[0]

    return None


def find_calories(lines: list[str], text: str):
    clean_text = fix_common_ocr_errors(text)

    kcal_matches = re.findall(
        r'(\d+(?:[.,]\d+)?)\s*(kcal|kkal)',
        clean_text,
        re.IGNORECASE,
    )

    for match in kcal_matches:
        value = to_float(match[0])

        if value is not None and 180 <= value <= 900 and value != 100:
            return int(value) if value.is_integer() else value

    for i, line in enumerate(lines):
        lower_line = fix_common_ocr_errors(line)

        if not any(word in lower_line for word in ['enerji', 'energy', 'kcal', 'kkal']):
            continue

        search_area = lines[max(0, i - 2) : min(i + 6, len(lines))]

        for current_line in search_area:
            current_lower = fix_common_ocr_errors(current_line)

            if is_invalid_candidate_line(current_line):
                continue

            numbers = extract_numbers(current_lower)

            candidates = [
                number
                for number in numbers
                if 180 <= number <= 900 and number != 100
            ]

            if candidates:
                value = candidates[-1]
                return int(value) if value.is_integer() else value

    return None


def find_fat_value(lines: list[str]):
    value = find_first_valid_number_near_label(
        lines,
        ['yag/fat', 'yağ/fat', 'yag', 'yağ', 'fat'],
        min_value=0,
        max_value=80,
        search_window=5,
        nutrient='fat',
    )

    if value is not None and value < 5:
        nearby_value = find_larger_value_near_label(
            lines,
            ['yag/fat', 'yağ/fat', 'yag', 'yağ', 'fat'],
            min_value=5,
            max_value=80,
            search_window=5,
            exclude_keywords=['doymus', 'doymuş', 'saturated', 'saturates'],
        )

        if nearby_value is not None:
            return nearby_value

    return value


def find_saturated_fat_value(lines: list[str]):
    return find_first_valid_number_near_label(
        lines,
        ['doymus yag', 'doymuş yağ', 'saturated', 'saturates'],
        min_value=0,
        max_value=60,
        search_window=6,
        nutrient='saturated_fat',
    )


def find_carb_value(lines: list[str]):
    return find_first_valid_number_near_label(
        lines,
        ['karbonhidrat', 'carbohydrate', 'carb'],
        min_value=0,
        max_value=100,
        search_window=5,
        nutrient='carb',
    )


def find_sugar_value(lines: list[str]):
    value = find_first_valid_number_near_label(
        lines,
        ['sekerler', 'şekerler', 'seker', 'şeker', 'sugars', 'sugar'],
        min_value=0,
        max_value=100,
        search_window=6,
        nutrient='sugar',
    )

    if value is not None and value < 5:
        nearby_value = find_larger_value_near_label(
            lines,
            ['sekerler', 'şekerler', 'seker', 'şeker', 'sugars', 'sugar'],
            min_value=5,
            max_value=100,
            search_window=6,
            exclude_keywords=['karbonhidrat', 'carbohydrate', 'carb'],
        )

        if nearby_value is not None:
            return nearby_value

    return value


def find_fiber_value(lines: list[str]):
    return find_first_valid_number_near_label(
        lines,
        ['lif', 'fiber', 'fibre'],
        min_value=0,
        max_value=40,
        search_window=4,
        nutrient='fiber',
    )


def find_protein_value(lines: list[str]):
    return find_first_valid_number_near_label(
        lines,
        ['protein'],
        min_value=0,
        max_value=60,
        search_window=5,
        nutrient='protein',
    )


def find_salt_value(lines: list[str]):
    return find_first_valid_number_near_label(
        lines,
        ['tuz', 'salt', 'salz'],
        min_value=0,
        max_value=10,
        search_window=4,
        nutrient='salt',
    )


def clean_number(value):
    if value is None:
        return None

    if isinstance(value, float) and value.is_integer():
        return int(value)

    return value


def extract_nutrition_data(text: str):
    fixed_text = fix_common_ocr_errors(text)
    lines = [line.strip() for line in fixed_text.split('\n') if line.strip()]

    ingredient_text = extract_ingredient_text(text)

    detected_allergens = detect_allergens(text)

    data = {
        'calories': clean_number(find_calories(lines, fixed_text)),
        'protein': clean_number(find_protein_value(lines)),
        'carb': clean_number(find_carb_value(lines)),
        'sugar': clean_number(find_sugar_value(lines)),
        'fat': clean_number(find_fat_value(lines)),
        'saturated_fat': clean_number(find_saturated_fat_value(lines)),
        'fiber': clean_number(find_fiber_value(lines)),
        'salt': clean_number(find_salt_value(lines)),
        'ingredients_text': ingredient_text,
        'detected_allergens': detected_allergens,
        'has_allergen_risk': len(detected_allergens) > 0,
    }

    return data