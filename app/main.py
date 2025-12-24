import uvicorn
from pathlib import Path
import json


def main(vicorn=None):
    """Основной запуск сервера"""
    print("🚦 Traffic Sign Analyzer")

    # Проверка моделей
    models = {
        "yolo": Path("../models/best.pt"),
        "resnet": Path("../models/best_state_classifier.pth")
    }

    for name, path in models.items():
        if path.exists():
            print(f"✅ {name}: найдена")
        else:
            print(f"❌ {name}: не найдена")

    # Создание директорий
    for folder in ["static/uploads", "static/results", "templates", "logs"]:
        Path(folder).mkdir(parents=True, exist_ok=True)

    # Создание файла логов если нет
    logs_file = Path("logs/detections.json")
    if not logs_file.exists():
        with open(logs_file, "w") as f:
            json.dump([], f)

    # Запуск сервера
    print("🌐 Сервер запущен: http://localhost:8000")
    uvicorn.run("app.routes:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()