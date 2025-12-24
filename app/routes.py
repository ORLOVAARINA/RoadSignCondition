from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from datetime import datetime
from pathlib import Path
import shutil
import sys
import os
# Добавляем родительскую директорию в путь для импорта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import SignDetector
app = FastAPI(title="Traffic Sign Analyzer")

# Настройки
BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Инициализация детектора
detector = None
@app.on_event("startup")
async def startup_event():
    """Инициализация детектора при запуске сервера"""
    global detector
    try:
        detector = SignDetector()
        print("✅ Детектор инициализирован успешно")
    except Exception as e:
        print(f"❌ Ошибка инициализации детектора: {e}")
        detector = None
# Маршруты
@app.get("/")
async def home(request: Request):
    logs = detector.get_logs() if detector else []
    return templates.TemplateResponse("index.html", {
        "request": request,
        "total_logs": len(logs),
        "last_log": logs[-1] if logs else None
    })


@app.get("/detect")
async def detect_page(request: Request):
    return templates.TemplateResponse("detect.html", {"request": request})


@app.post("/detect")
async def detect_image(
        request: Request,
        file: UploadFile = File(...),
        confidence: float = Form(0.5)
):
    if not detector:
        return templates.TemplateResponse("detect.html", {
            "request": request,
            "error": "Модели не загружены"
        })

    # Проверка типа файла
    if not file.content_type.startswith("image/"):
        return templates.TemplateResponse("detect.html", {
            "request": request,
            "error": "Загрузите изображение (JPG, PNG, etc.)"
        })

    # Сохранение файла
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    filepath = f"static/uploads/{filename}"

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Детекция
    results = detector.detect(filepath, confidence)

    # Визуализация
    result_path = f"static/results/result_{timestamp}.jpg"
    detector.draw_results(filepath, results, result_path)

    return templates.TemplateResponse("detect.html", {
        "request": request,
        "results": results,
        "original": f"/static/uploads/{filename}",
        "result": f"/static/results/result_{timestamp}.jpg"
    })


@app.get("/logs")
async def logs_page(request: Request):
    logs = detector.get_logs() if detector else []
    reversed_logs = list(reversed(logs))
    return templates.TemplateResponse("logs.html", {
        "request": request,
        "logs": reversed_logs,
        "total_logs": len(logs)
    })
