import unittest

from app.services.parser import parse_nutrition_from_text


class NutritionParserExamplesTest(unittest.TestCase):
    def test_turkish_label_with_allergens(self):
        text = """Enerji 1883 kJ / 450 kcal
Yağ 18 g
Doymuş yağ 7,5 g
Karbonhidrat 62 g
Şekerler 28 g
Protein 7,2 g
Lif 3 g
Tuz 0,35 g
İçindekiler: süt, soya, gluten, fındık içerebilir."""

        result = parse_nutrition_from_text(text)

        self.assertEqual(result['calories'], 450)
        self.assertEqual(result['fat'], 18)
        self.assertEqual(result['saturated_fat'], 7.5)
        self.assertEqual(result['carbs'], 62)
        self.assertEqual(result['sugar'], 28)
        self.assertEqual(result['protein'], 7.2)
        self.assertEqual(result['fiber'], 3)
        self.assertEqual(result['salt'], 0.35)
        self.assertIn('MILK', result['detected_allergens'])
        self.assertIn('SOY', result['detected_allergens'])
        self.assertIn('GLUTEN', result['detected_allergens'])
        self.assertIn('HAZELNUT', result['detected_allergens'])

    def test_english_label_with_joined_units(self):
        text = """Energy 1200 kJ 286 kcal
Fat 12g
Saturated Fat 4.5g
Carbohydrate 33g
Sugars 15g
Protein 6g
Salt 0.8g
Contains milk and wheat."""

        result = parse_nutrition_from_text(text)

        self.assertEqual(result['calories'], 286)
        self.assertEqual(result['fat'], 12)
        self.assertEqual(result['saturated_fat'], 4.5)
        self.assertEqual(result['carbs'], 33)
        self.assertEqual(result['sugar'], 15)
        self.assertEqual(result['protein'], 6)
        self.assertEqual(result['salt'], 0.8)
        self.assertIn('MILK', result['detected_allergens'])
        self.assertIn('WHEAT', result['detected_allergens'])

    def test_window_based_values(self):
        text = """Karbonhidrat
55 g
Şekerler
24 g
Protein
8 g
Yağ
16 g"""

        result = parse_nutrition_from_text(text)

        self.assertEqual(result['carbs'], 55)
        self.assertEqual(result['sugar'], 24)
        self.assertEqual(result['protein'], 8)
        self.assertEqual(result['fat'], 16)


if __name__ == '__main__':
    unittest.main()
