import unittest

from app.services.parser import parse_nutrition_from_boxes


def make_box(text: str, x: float, y: float, width: float = 120, height: float = 20):
    return {
        'text': text,
        'conf': 0.9,
        'bbox': [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
        'x_min': x,
        'x_max': x + width,
        'y_min': y,
        'y_max': y + height,
        'x_center': x + width / 2,
        'y_center': y + height / 2,
        'width': width,
        'height': height,
    }


class NutritionBoxParserTest(unittest.TestCase):
    def test_clean_table_rows(self):
        rows = [
            ('Enerji ve Besin Öğeleri', '100 g'),
            ('Enerji', '122 kcal'),
            ('Yağ', '8.0 g'),
            ('Doymuş yağ', '2.0 g'),
            ('Karbonhidrat', '62 g'),
            ('Şekerler', '50 g'),
            ('Lif', '6.0 g'),
            ('Protein', '11 g'),
            ('Tuz', '0.03 g'),
        ]
        boxes = []

        for index, (label, value) in enumerate(rows):
            y = index * 30
            boxes.append(make_box(label, 20, y, 180))
            boxes.append(make_box(value, 280, y, 90))

        result = parse_nutrition_from_boxes('\n'.join(f'{label} {value}' for label, value in rows), boxes)

        self.assertEqual(result['calories'], 122)
        self.assertEqual(result['fat'], 8)
        self.assertEqual(result['saturated_fat'], 2)
        self.assertEqual(result['carbs'], 62)
        self.assertEqual(result['sugar'], 50)
        self.assertEqual(result['fiber'], 6)
        self.assertEqual(result['protein'], 11)
        self.assertEqual(result['salt'], 0.03)

    def test_noisy_rows_do_not_randomly_assign_values(self):
        tokens = [
            'Enej',
            '12E2',
            'Yal',
            '80',
            'Dms ya',
            '20',
            'Kbonidal',
            '62',
            'fkene',
            '60',
            'Proicn',
            '11',
            'Iu',
            '003',
        ]
        boxes = [
            make_box(text, 20 if index % 2 == 0 else 280, index * 30)
            for index, text in enumerate(tokens)
        ]
        result = parse_nutrition_from_boxes('\n'.join(tokens), boxes)

        self.assertEqual(result['calories'], 122)
        self.assertEqual(result['fat'], 8)
        self.assertEqual(result['saturated_fat'], 2)
        self.assertEqual(result['carbs'], 62)
        self.assertIsNone(result['sugar'])
        self.assertEqual(result['protein'], 11)
        self.assertEqual(result['salt'], 0.03)

    def test_ingredients_do_not_feed_nutrition_numbers(self):
        text = (
            'Ingredients: Dates (75%), Peanuis (10%), Cocoa Mass...\n'
            'Allergens: Contains Hazehnuts and Peanuts. May contain Almonds, '
            'Pistachios, Cashews and Milk Products.'
        )
        boxes = [
            make_box('Ingredients: Dates (75%), Peanuis (10%), Cocoa Mass...', 20, 0, 460),
            make_box('Allergens: Contains Hazehnuts and Peanuts.', 20, 30, 420),
            make_box('May contain Almonds, Pistachios, Cashews and Milk Products.', 20, 60, 520),
        ]
        result = parse_nutrition_from_boxes(text, boxes)

        for field in ['calories', 'fat', 'saturated_fat', 'carbs', 'sugar', 'protein', 'fiber', 'salt']:
            self.assertIsNone(result[field])

        for allergen in ['HAZELNUT', 'PEANUT', 'ALMOND', 'PISTACHIO', 'CASHEW', 'MILK']:
            self.assertIn(allergen, result['detected_allergens'])


if __name__ == '__main__':
    unittest.main()
