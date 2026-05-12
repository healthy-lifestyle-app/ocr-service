from fastapi import APIRouter, File, UploadFile, HTTPException
from app.services.ocr_engine import extract_text_from_image
from app.services.parser import extract_nutrition_data
import traceback


router = APIRouter()


@router.post('/extract')
async def extract_ocr(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='Lütfen bir görsel yükleyin.')

    try:
        image_bytes = await file.read()
        ocr_result = extract_text_from_image(image_bytes)

        raw_text = (
            ocr_result.get('raw_text')
            or ocr_result.get('text')
            or ''
        ).strip()

        lines = ocr_result.get('lines') or []

        if not raw_text:
            return {
                'success': False,
                'message': 'OCR metni okunamadı.',
                'raw_text': '',
                'text': '',
                'lines': [],
                'calories': None,
                'protein': None,
                'carb': None,
                'carbs': None,
                'sugar': None,
                'fat': None,
                'saturated_fat': None,
                'fiber': None,
                'salt': None,
                'ingredients_text': '',
                'detected_allergens': [],
                'has_allergen_risk': False,
            }

        parser_error = False

        try:
            nutrition_data = extract_nutrition_data(raw_text)
        except Exception as parser_exception:
            parser_error = True
            nutrition_data = {}
            print('[PARSER] error:', str(parser_exception))
            traceback.print_exc()

        carb_value = nutrition_data.get('carb')
        carbs_value = nutrition_data.get('carbs')

        if carb_value is None:
            carb_value = carbs_value

        if carbs_value is None:
            carbs_value = carb_value

        detected_allergens = nutrition_data.get('detected_allergens') or []

        response = {
            'success': True,
            'raw_text': raw_text,
            'text': raw_text,
            'lines': lines,
            'calories': nutrition_data.get('calories'),
            'protein': nutrition_data.get('protein'),
            'carb': carb_value,
            'carbs': carbs_value,
            'sugar': nutrition_data.get('sugar'),
            'fat': nutrition_data.get('fat'),
            'saturated_fat': nutrition_data.get('saturated_fat'),
            'fiber': nutrition_data.get('fiber'),
            'salt': nutrition_data.get('salt'),
            'ingredients_text': nutrition_data.get('ingredients_text') or '',
            'detected_allergens': detected_allergens,
            'has_allergen_risk': bool(
                nutrition_data.get('has_allergen_risk')
                or len(detected_allergens) > 0
            ),
        }

        if parser_error:
            response['message'] = (
                'OCR metni okundu ancak bazı besin değerleri çıkarılamadı.'
            )

        print(f'[PARSER] keys={list(nutrition_data.keys())}')
        print(f'[PARSER] allergens={detected_allergens}')

        return response

    except ValueError as e:
        print('ValueError:', str(e))
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        print('GENEL HATA:', str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f'OCR hatası: {str(e)}')