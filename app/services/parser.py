import re
ALLERGEN_KEYWORDS = {
    'GLUTEN': ['gluten', 'gluten içeren tahıllar', 'gluten iceren tahillar'],
    'WHEAT': ['wheat', 'buğday', 'bugday', 'buğday unu', 'bugday unu', 'wheat flour'],
    'RYE': ['rye', 'çavdar', 'cavdar'],
    'BARLEY': ['barley', 'arpa', 'malt', 'arpa malt'],
    'OATS': ['oats', 'yulaf'],
    'SPELT': ['spelt'],
    'KAMUT': ['kamut'],

    'EGG': ['egg', 'yumurta'],
    'EGG_WHITE': ['egg white', 'yumurta akı', 'yumurta aki'],
    'EGG_YOLK': ['egg yolk', 'yumurta sarısı', 'yumurta sarisi'],
    'ALBUMIN': ['albumin', 'albümin', 'albumin'],
    'OVALBUMIN': ['ovalbumin', 'ovalbümin'],
    'LYSOZYME': ['lysozyme', 'lizozim'],

    'FISH': ['fish', 'balık', 'balik'],
    'SALMON': ['salmon', 'somon'],
    'TUNA': ['tuna', 'ton balığı', 'ton baligi'],
    'COD': ['cod', 'morina'],
    'ANCHOVY': ['anchovy', 'hamsi', 'ançüez', 'ancuez'],
    'FISH_GELATIN': ['fish gelatin', 'balık jelatini', 'balik jelatini'],
    'ISINGLASS': ['isinglass'],

    'PEANUT': ['peanut', 'yer fıstığı', 'yer fistigi'],
    'SOY': ['soy', 'soya'],
    'SOY_PROTEIN': ['soy protein', 'soya proteini'],
    'SOY_FLOUR': ['soy flour', 'soya unu'],
    'SOY_LECITHIN': ['soy lecithin', 'soya lesitini', 'lesitin', 'lecithin'],

    'MILK': ['milk', 'süt', 'sut', 'süt ürünü', 'sut urunu', 'süt ürünleri', 'sut urunleri'],
    'CASEIN': ['casein', 'kazein'],
    'CASEINATE': ['caseinate', 'kazeinat'],
    'WHEY': ['whey', 'peynir altı suyu', 'peynir alti suyu'],
    'LACTALBUMIN': ['lactalbumin', 'laktalbumin'],
    'LACTOGLOBULIN': ['lactoglobulin', 'laktoglobulin'],
    'LACTOSE': ['lactose', 'laktoz'],

    'TREE_NUTS': ['tree nuts', 'sert kabuklu yemiş', 'sert kabuklu yemis'],
    'ALMOND': ['almond', 'badem'],
    'HAZELNUT': ['hazelnut', 'hazalnut', 'fındık', 'findik'],
    'WALNUT': ['walnut', 'ceviz'],
    'CASHEW': ['cashew', 'kaju'],
    'PECAN': ['pecan', 'pekan cevizi'],
    'BRAZIL_NUT': ['brazil nut', 'brezilya cevizi'],
    'PISTACHIO': ['pistachio', 'pisachio', 'antep fıstığı', 'antep fistigi'],
    'MACADAMIA': ['macadamia', 'makademya'],

    'CELERY': ['celery', 'kereviz'],
    'MUSTARD': ['mustard', 'hardal'],
    'SESAME': ['sesame', 'susam'],
    'SULFITES': ['sulfites', 'sulphites', 'sülfit', 'sulfit', 'kükürt dioksit', 'kukurt dioksit'],
    'LUPIN': ['lupin', 'acı bakla', 'aci bakla'],

    'CRUSTACEANS': ['crustaceans', 'kabuklular'],
    'SHRIMP': ['shrimp', 'karides'],
    'PRAWN': ['prawn', 'jumbo karides'],
    'CRAB': ['crab', 'yengeç', 'yengec'],
    'LOBSTER': ['lobster', 'ıstakoz', 'istakoz'],
    'CRAYFISH': ['crayfish', 'tatlı su kereviti', 'tatli su kereviti'],

    'MOLLUSCS': ['molluscs', 'mollusks', 'yumuşakçalar', 'yumusakcakar', 'yumuşakcalar'],
    'MUSSEL': ['mussel', 'midye'],
    'OYSTER': ['oyster', 'istiridye'],
    'SQUID': ['squid', 'kalamar'],
    'OCTOPUS': ['octopus', 'ahtapot'],
    'CLAM': ['clam', 'kum midyesi'],
    'SCALLOP': ['scallop', 'deniz tarağı', 'deniz taragi'],

    'COCONUT': ['coconut', 'hindistan cevizi'],
    'CORN': ['corn', 'mısır', 'misir'],
    'BUCKWHEAT': ['buckwheat', 'karabuğday', 'karabugday'],
    'CHICKPEA': ['chickpea', 'nohut'],
    'LENTIL': ['lentil', 'mercimek'],
    'PEA_PROTEIN': ['pea protein', 'bezelye proteini'],
    'SUNFLOWER_SEED': ['sunflower seed', 'ay çekirdeği', 'ay cekirdegi'],
    'POPPY_SEED': ['poppy seed', 'haşhaş tohumu', 'hashas tohumu'],

    'CARMINE_E120': ['carmine', 'cochineal', 'e120', 'karmin', 'koşinil', 'kosinil'],
    'ANNATTO': ['annatto', 'achiote'],
    'KIWI': ['kiwi', 'kivi'],
    'STRAWBERRY': ['strawberry', 'çilek', 'cilek'],
    'TOMATO': ['tomato', 'domates'],
    'PEACH': ['peach', 'şeftali', 'seftali'],
    'APPLE': ['apple', 'elma'],
    'BANANA': ['banana', 'muz'],
    'AVOCADO': ['avocado', 'avokado'],
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


def extract_ingredient_text(text: str):
    lower_text = normalize_text(text)

    start_keywords = [
        'icindekiler',
        'ingredients',
        'allergen information',
        'alerjen',
        'contains',
        'may contain',
    ]

    for keyword in start_keywords:
        index = lower_text.find(keyword)

        if index != -1:
            return text[index:]

    return text


def detect_allergens(text: str):
    normalized_text = normalize_text(text)
    detected = []

    for allergen_code, keywords in ALLERGEN_KEYWORDS.items():
        for keyword in keywords:
            normalized_keyword = normalize_text(keyword)

            pattern = rf'(?<![a-zA-Z]){re.escape(normalized_keyword)}(?![a-zA-Z])'

            if re.search(pattern, normalized_text):
                detected.append(allergen_code)
                break

    return detected

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


def find_value_after_keyword(
    lines: list[str],
    keywords: list[str],
    max_lookahead: int = 6,
    max_valid_value: float | None = None,
):
    skip_words = [
        'prod.code',
        'prod code',
        'cesit no',
        'çeşit no',
        'registration',
        'kayit',
        'kayıt',
        'ref.no',
        'relno',
        'bsc',
        'tr-',
        'barcode',
        'www',
        'http',
        'tel',
    ]

    for i, line in enumerate(lines):
        lower_line = line.lower()

        if any(keyword in lower_line for keyword in keywords):
            for j in range(i, min(i + max_lookahead + 1, len(lines))):
                current_line = lines[j].lower()

                if any(skip_word in current_line for skip_word in skip_words):
                    continue

                numbers = extract_numbers(lines[j])

                valid_numbers = []

                for number in numbers:
                    if max_valid_value is not None and number > max_valid_value:
                        continue

                    valid_numbers.append(number)

                if valid_numbers:
                    return valid_numbers[0]

    return None


def find_protein_value(lines: list[str]):
    for i, line in enumerate(lines):
        if 'protein' in line.lower():
            for j in range(i + 1, min(i + 7, len(lines))):
                current_line = lines[j].lower()

                if any(
                    word in current_line
                    for word in ['prod.code', 'prod code', 'cesit no', 'çeşit no']
                ):
                    continue

                numbers = extract_numbers(lines[j])

                for number in numbers:
                    if 0 <= number <= 100:
                        return number

    return None


def find_fat_value(lines: list[str]):
    for i, line in enumerate(lines):
        lower_line = line.lower()

        if (
    'yag' in lower_line and 'fat' in lower_line
) or (
    'yağ' in lower_line and 'fat' in lower_line
) or 'fat/fett' in lower_line:
            for j in range(i + 1, min(i + 4, len(lines))):
                numbers = extract_numbers(lines[j])

                for number in numbers:
                    if 0 <= number <= 100:
                        return number

    return None


def extract_nutrition_data(text: str):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    clean_text = text.lower()
    
    ingredient_text = extract_ingredient_text(text)
    detected_allergens = detect_allergens(ingredient_text) 
    
    data = {
        'calories': None,
        'protein': None,
        'carb': None,
        'sugar': None,
        'fat': None,
        'saturated_fat': None,
        'fiber': None,
        'salt': None,
        'ingredients_text': ingredient_text,
        'detected_allergens': detected_allergens,
        'has_allergen_risk': len(detected_allergens) > 0,
    }

    calorie_match = re.search(
    r'(\d+(?:[.,]\d+)?)\s*kcal\)?\s*/?\s*100\s*g',
    clean_text,
    re.IGNORECASE,
)

    if calorie_match:
        data['calories'] = to_float(calorie_match.group(1))
    else:
        energy_index = None

        for i, line in enumerate(lines):
            lower_line = line.lower()
            if 'enerji' in lower_line or 'energy' in lower_line or 'calories' in lower_line:
                energy_index = i
                break

        if energy_index is not None:
            for j in range(energy_index + 1, min(energy_index + 5, len(lines))):
                numbers = extract_numbers(lines[j])

                kcal_candidates = [number for number in numbers if 100 <= number <= 900]

                if kcal_candidates:
                    data['calories'] = kcal_candidates[-1]
                    break

    data['fat'] = find_fat_value(lines)

    data['saturated_fat'] = find_value_after_keyword(
        lines,
        ['doymus yag', 'doymuş yağ', 'saturates', 'saturated fat'],
        max_valid_value=50,
    )

    data['carb'] = find_value_after_keyword(
        lines,
        ['karbonhidrat', 'carbohydrate', 'carb'],
        max_valid_value=100,
    )

    data['sugar'] = find_value_after_keyword(
        lines,
        ['sekerler', 'şekerler', 'sugars', 'sugar'],
        max_valid_value=100,
    )

    data['fiber'] = find_value_after_keyword(
        lines,
        ['lif', 'fibre', 'fiber'],
        max_valid_value=100,
    )

    data['protein'] = find_protein_value(lines)

    data['salt'] = find_value_after_keyword(
        lines,
        ['tuz', 'salt', 'sall', 'salz'],
        max_valid_value=100,
    )

    if data['fiber'] is not None and data['fiber'] > 20:
        data['fiber'] = data['fiber'] / 10

    for key in ['protein', 'carb', 'sugar', 'fat', 'saturated_fat', 'fiber', 'salt']:
        if data[key] is not None and data[key] > 100:
            data[key] = None

    return data