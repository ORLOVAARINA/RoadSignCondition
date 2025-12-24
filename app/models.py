import torch
import cv2
from PIL import Image
from torch import nn
from ultralytics import YOLO
from torchvision import models, transforms
import json
from datetime import datetime
from pathlib import Path
import sys
import os


class SignDetector:
    def __init__(self):
        """Инициализация детектора"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Получаем корневую директорию проекта
        self.root_dir = Path(__file__).parent.parent

        # Загрузка моделей с правильными путями
        self.yolo = YOLO(str(self.root_dir / 'models' / 'best.pt'))  # Изменено здесь
        self.resnet = self.load_classifier(str(self.root_dir / 'models' / 'best_state_classifier.pth'))

        # Трансформации
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Классы (убедитесь, что они совпадают с вашим датасетом)
        self.sign_names = ['bus_stop', 'do_not_enter', 'do_not_stop', 'do_not_turn_l',
                           'do_not_turn_r', 'do_not_u_turn', 'enter_left_lane', 'green_light',
                           'left_right_lane', 'no_parking', 'parking', 'ped_crossing',
                           'ped_zebra_cross', 'railway_crossing', 'red_light', 'stop',
                           't_intersection_l', 'traffic_light', 'u_turn', 'warning', 'yellow_light']

        self.conditions = ['damaged', 'dirty', 'normal']
        self.condition_colors = {
            'normal': '#10b981',
            'damaged': '#ef4444',
            'dirty': '#f59e0b',
            'unknown': '#6b7280'
        }

        print(f"✅ Детектор инициализирован на устройстве: {self.device}")

    def load_classifier(self, path):
        """Загрузка классификатора"""
        # Загружаем checkpoint
        checkpoint = torch.load(path, map_location=self.device)

        # Создаем модель с ТОЙ ЖЕ архитектурой, что и при обучении
        model = models.resnet50(pretrained=False)  # Не загружаем ImageNet веса!

        # ВАЖНО: Повторяем архитектуру из кода обучения:
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)  # 3 класса: normal, damaged, dirty
        )

        # Извлекаем state_dict модели из checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Если это уже сам state_dict
            state_dict = checkpoint

        # Убираем префикс 'module.' если модель была сохранена как DataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Убираем 'module.'
            else:
                new_state_dict[k] = v

        # Загружаем веса
        model.load_state_dict(new_state_dict)

        model.to(self.device)
        model.eval()
        return model

    def detect(self, image_path, confidence=0.5):
        """Основная функция детекции"""
        try:
            start_time = datetime.now()

            # Проверяем существование файла
            if not Path(image_path).exists():
                return {
                    'success': False,
                    'error': f'Файл не найден: {image_path}',
                    'signs': [],
                    'total': 0,
                    'processing_time': 0
                }

            # Загрузка изображения
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'success': False,
                    'error': f'Не удалось загрузить изображение: {image_path}',
                    'signs': [],
                    'total': 0,
                    'processing_time': 0
                }

            results = []

            # YOLO детекция
            yolo_results = self.yolo(image, conf=confidence)[0]

            if yolo_results.boxes is not None and len(yolo_results.boxes) > 0:
                for box, cls, conf in zip(yolo_results.boxes.xyxy,
                                          yolo_results.boxes.cls,
                                          yolo_results.boxes.conf):
                    x1, y1, x2, y2 = map(int, box)

                    # Проверяем индекс класса
                    cls_idx = int(cls)
                    if cls_idx >= len(self.sign_names):
                        sign_type = f"unknown_{cls_idx}"
                    else:
                        sign_type = self.sign_names[cls_idx]

                    # Классификация состояния
                    crop = image[y1:y2, x1:x2]
                    condition, cond_conf = self.classify_condition(crop)

                    results.append({
                        'bbox': [x1, y1, x2, y2],
                        'type': sign_type,
                        'detection_conf': float(conf),
                        'condition': condition,
                        'condition_conf': float(cond_conf),
                        'color': self.condition_colors.get(condition, '#6b7280')
                    })

            processing_time = (datetime.now() - start_time).total_seconds()

            # Сохранение в логи
            self.save_log(image_path, results, processing_time)

            return {
                'success': True,
                'signs': results,
                'total': len(results),
                'processing_time': round(processing_time, 2)
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'signs': [],
                'total': 0,
                'processing_time': 0
            }

    def classify_condition(self, crop):
        """Классификация состояния знака"""
        try:
            if crop.shape[0] == 0 or crop.shape[1] == 0:
                return 'unknown', 0.0

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(crop_rgb)
            tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.resnet(tensor)
                prob = torch.softmax(output, dim=1)
                conf, pred = torch.max(prob, 1)

            condition_idx = pred.item()
            if condition_idx >= len(self.conditions):
                return 'unknown', 0.0

            return self.conditions[condition_idx], conf.item()
        except Exception as e:
            print(f"Ошибка классификации: {e}")
            return 'unknown', 0.0

    def draw_results(self, image_path, results, output_path):
        """Отрисовка результатов на изображении"""
        try:
            if 'error' in results:
                return None

            image = cv2.imread(image_path)
            if image is None:
                return None

            for sign in results.get('signs', []):
                x1, y1, x2, y2 = sign['bbox']

                # Конвертируем hex цвет в BGR
                hex_color = sign.get('color', '#6b7280').lstrip('#')
                if len(hex_color) == 6:
                    b = int(hex_color[4:6], 16)
                    g = int(hex_color[2:4], 16)
                    r = int(hex_color[0:2], 16)
                    bgr_color = (b, g, r)
                else:
                    bgr_color = (128, 128, 128)  # серый

                # Рисуем прямоугольник
                cv2.rectangle(image, (x1, y1), (x2, y2), bgr_color, 3)

                # Текст
                label = f"{sign['type']} - {sign['condition']} ({sign['detection_conf']:.2f})"
                cv2.putText(image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr_color, 2)

            cv2.imwrite(output_path, image)
            return output_path

        except Exception as e:
            print(f"Ошибка отрисовки: {e}")
            return None

    def save_log(self, image_path, results, processing_time):
        """Сохранение результата в логи"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'image': Path(image_path).name,
                'processing_time': round(processing_time, 2),
                'total_signs': len(results),
                'results': results
            }

            logs_dir = self.root_dir / "logs"
            logs_dir.mkdir(exist_ok=True)

            logs_file = logs_dir / "detections.json"
            logs = []

            if logs_file.exists():
                try:
                    with open(logs_file, 'r') as f:
                        logs = json.load(f)
                except:
                    logs = []

            logs.append(log_entry)

            # Сохраняем только последние 50 записей
            with open(logs_file, 'w') as f:
                json.dump(logs[-50:], f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Ошибка сохранения лога: {e}")

    def get_logs(self):
        """Получение всех логов"""
        try:
            logs_file = self.root_dir / "logs" / "detections.json"
            if logs_file.exists():
                with open(logs_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"Ошибка чтения логов: {e}")
            return []