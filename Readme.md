# ocr-service

FastAPI tabanlı basit OCR servisi.

## Kurulum

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install fastapi uvicorn python-multipart numpy==1.26.4 opencv-python==4.6.0.66 paddlepaddle==2.6.2 paddleocr==2.7.3