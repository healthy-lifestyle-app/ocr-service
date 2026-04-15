from fastapi import APIRouter, File, UploadFile, HTTPException
from app.services.ocr_engine import extract_text_from_image
import traceback

router = APIRouter()

@router.post('/extract')
async def extract_ocr(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='Lütfen bir görsel yükleyin.')

    try:
        image_bytes = await file.read()
        result = extract_text_from_image(image_bytes)
        return result
    except ValueError as e:
        print('ValueError:', str(e))
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print('GENEL HATA:', str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f'OCR hatası: {str(e)}')