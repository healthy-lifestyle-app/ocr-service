from fastapi import FastAPI
from app.routes.ocr import router as ocr_router

app = FastAPI(title='OCR API')

app.include_router(ocr_router, prefix='/ocr', tags=['OCR'])

@app.get('/')
def root():
    return {'message': 'OCR API çalışıyor'}