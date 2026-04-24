import csv
import os


CSV_COLUMNS = [
    'calories',
    'protein',
    'carb',
    'sugar',
    'fat',
    'saturated_fat',
    'fiber',
    'salt',
    'detected_allergens',
    'has_allergen_risk',
]


def save_to_csv(data: dict, filename: str = 'data/food_data.csv'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    csv_data = {
        'calories': data.get('calories'),
        'protein': data.get('protein'),
        'carb': data.get('carb'),
        'sugar': data.get('sugar'),
        'fat': data.get('fat'),
        'saturated_fat': data.get('saturated_fat'),
        'fiber': data.get('fiber'),
        'salt': data.get('salt'),
        'detected_allergens': ','.join(data.get('detected_allergens', [])),
        'has_allergen_risk': data.get('has_allergen_risk'),
    }

    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)

        if not file_exists:
            writer.writeheader()

        writer.writerow(csv_data)

    print('CSV satırı yazıldı:', csv_data)