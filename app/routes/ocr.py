from fastapi import APIRouter, File, UploadFile, HTTPException
from app.services.ocr_engine import extract_text_from_image
from app.services.parser import extract_nutrition_data
from app.services.csv_writer import save_to_csv
import traceback

router = APIRouter()

@router.post('/extract')
async def extract_ocr(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='Lütfen bir görsel yükleyin.')

    try:
        image_bytes = await file.read()

        result = extract_text_from_image(image_bytes)

        parsed = extract_nutrition_data(result['text'])

        save_to_csv(parsed)

        return {
            'raw_text': result['text'],
            'lines': result['lines'],
            'parsed': parsed,
        }

    except ValueError as e:
        print('ValueError:', str(e))
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print('GENEL HATA:', str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f'OCR hatası: {str(e)}')