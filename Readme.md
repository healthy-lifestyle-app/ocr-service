# ocr-service

FastAPI tabanlı OCR servisi. OCR motoru CPU modunda EasyOCR kullanır ve
Türkçe/İngilizce metin okur.

## Mac kurulumu

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Test

```bash
curl -X POST http://127.0.0.1:8000/ocr/extract \
  -F "file=@/Users/melisayasarturk/Downloads/IMG_0169.jpeg"
```

Parser smoke test:

```bash
python -m unittest tests.test_parser_examples
python -m unittest tests.test_nutrition_parser
```

Debug response:

```bash
curl -X POST "http://127.0.0.1:8000/ocr/extract?debug=true" \
  -F "file=@/Users/melisayasarturk/Downloads/IMG_0169.jpeg"
```

Parser smoke test:

```bash
python -m unittest tests.test_parser_examples
python -m unittest tests.test_nutrition_parser
```

Debug response:

```bash
curl -X POST "http://127.0.0.1:8000/ocr/extract?debug=true" \
  -F "file=@/Users/melisayasarturk/Downloads/IMG_0169.jpeg"
```

Beklenen response:

```json
{
  "raw_text": "...",
  "lines": ["...", "..."],
  "parsed": {
    "calories": 332,
    "protein": 6.2,
    "carb": 62,
    "sugar": null,
    "fat": 9,
    "saturated_fat": null,
    "fiber": null,
    "salt": null,
    "ingredients_text": "...",
    "detected_allergens": ["PEANUT"],
    "has_allergen_risk": true
  },
  "nutrition": {
    "calories": 332,
    "protein": 6.2,
    "carbs": 62
  },
  "ocr_debug": {
    "variant": "gray_resize_2x",
    "score": 42.1
  },
  "nutrition": {
    "calories": 332,
    "protein": 6.2,
    "carbs": 62
  },
  "ocr_debug": {
    "variant": "gray_resize_2x",
    "score": 42.1
  }
}
```
