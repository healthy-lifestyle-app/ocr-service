import csv
import os
from typing import Any


CSV_COLUMNS = [
    'product_name',
    'calories',
    'protein',
    'carb',
    'sugar',
    'fat',
    'saturated_fat',
    'fiber',
    'salt',
    'contains_milk',
    'contains_soy',
    'contains_gluten',
    'contains_wheat',
    'contains_peanut',
    'contains_hazelnut',
    'contains_almond',
    'contains_walnut',
    'contains_cashew',
    'contains_pistachio',
    'contains_egg',
    'contains_fish',
    'contains_sesame',
    'contains_mustard',
    'contains_celery',
    'contains_sulfites',
    'contains_lupin',
    'contains_crustaceans',
    'contains_molluscs',
    'detected_allergens',
    'has_allergen_risk',
]


ALLERGEN_FLAG_COLUMNS = {
    'MILK': 'contains_milk',
    'SOY': 'contains_soy',
    'GLUTEN': 'contains_gluten',
    'WHEAT': 'contains_wheat',
    'PEANUT': 'contains_peanut',
    'HAZELNUT': 'contains_hazelnut',
    'ALMOND': 'contains_almond',
    'WALNUT': 'contains_walnut',
    'CASHEW': 'contains_cashew',
    'PISTACHIO': 'contains_pistachio',
    'EGG': 'contains_egg',
    'FISH': 'contains_fish',
    'SESAME': 'contains_sesame',
    'MUSTARD': 'contains_mustard',
    'CELERY': 'contains_celery',
    'SULFITES': 'contains_sulfites',
    'LUPIN': 'contains_lupin',
    'CRUSTACEAN': 'contains_crustaceans',
    'CRUSTACEANS': 'contains_crustaceans',
    'MOLLUSCS': 'contains_molluscs',
}


def _clean_value(value: Any):
    if value is None:
        return ''

    if isinstance(value, bool):
        return int(value)

    return value


def _build_allergen_flags(detected_allergens: list[str]) -> dict:
    normalized_allergens = {
        str(allergen).upper().strip()
        for allergen in detected_allergens
        if allergen
    }

    flags = {}

    for allergen_code, column_name in ALLERGEN_FLAG_COLUMNS.items():
        flags[column_name] = int(allergen_code in normalized_allergens)

    return flags


def save_to_csv(
    data: dict,
    product_name: str = 'Unknown Product',
    filename: str = 'data/food_data.csv',
):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    detected_allergens = data.get('detected_allergens') or []

    if not isinstance(detected_allergens, list):
        detected_allergens = []

    allergen_flags = _build_allergen_flags(detected_allergens)

    csv_data = {
        'product_name': product_name,
        'calories': _clean_value(data.get('calories')),
        'protein': _clean_value(data.get('protein')),
        'carb': _clean_value(data.get('carb')),
        'sugar': _clean_value(data.get('sugar')),
        'fat': _clean_value(data.get('fat')),
        'saturated_fat': _clean_value(data.get('saturated_fat')),
        'fiber': _clean_value(data.get('fiber')),
        'salt': _clean_value(data.get('salt')),
        **allergen_flags,
        'detected_allergens': ','.join(detected_allergens),
        'has_allergen_risk': int(bool(data.get('has_allergen_risk'))),
    }

    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)

        if not file_exists:
            writer.writeheader()

        writer.writerow(csv_data)
