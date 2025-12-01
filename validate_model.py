#!/usr/bin/env python3
"""
Валідація моделі на стабільному тестовому зразку
"""

import argparse
import json
import os
import warnings

import torch
import numpy as np

from model_utils import SmallCNN, wav_to_melspec, load_model, get_model_metadata

warnings.filterwarnings("ignore", category=UserWarning)


def validate_model_consistency(model_path: str):
    """
    Перевіряє консистентність моделі на стабільному зразку
    """
    print("=" * 60)
    print("🔍 ВАЛІДАЦІЯ МОДЕЛІ")
    print("=" * 60)

    # Перевірка існування моделі
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не знайдена: {model_path}")

    print(f"✅ Модель знайдена: {model_path}")
    print(f"📏 Розмір моделі: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")

    # Читаємо метадані моделі
    print(f"📖 Читання метаданих моделі...")
    metadata = get_model_metadata(model_path)
    n_classes = metadata['n_classes']
    class_names = metadata['class_names']

    if n_classes is None:
        raise ValueError("Не вдалося визначити кількість класів з моделі. Модель може бути пошкоджена.")

    print(f"✅ Метадані зчитано:")
    print(f"   - Кількість класів: {n_classes}")
    print(f"   - Назви класів: {'Так' if class_names else 'Ні'}")

    # Використовуємо збережені назви класів або створюємо generic
    if class_names:
        CLASSES = class_names
    else:
        CLASSES = [f"class_{i}" for i in range(n_classes)]
        print(f"⚠️  Назви класів не знайдені, використовуємо generic: {CLASSES}")

    # Завантаження моделі (n_classes буде взято з метаданих)
    device = torch.device("cpu")
    model = load_model(SmallCNN, model_path, n_classes=None, device=device)
    model.eval()

    print("✅ Модель завантажена успішно")

    # Створення тестового зразка
    waveform, sample_rate = create_stable_test_sample()
    print(f"✅ Тестовий зразок створено: {waveform.shape}")

    # Перевірка моделі 5 разів для консистентності
    predictions = []
    confidences = []

    print("\n🧪 Тестування консистентності (5 ітерацій)...")

    with torch.no_grad():
        for i in range(5):
            melspec = wav_to_melspec(waveform, sample_rate).unsqueeze(0).to(device)
            output = model(melspec)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()

            predictions.append(predicted_class)
            confidences.append(confidence)

            print(f"  Ітерація {i+1}: клас={CLASSES[predicted_class]}, довіра={confidence:.4f}")

    # Перевірка консистентності
    is_consistent = len(set(predictions)) == 1
    avg_confidence = np.mean(confidences)
    std_confidence = np.std(confidences)

    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТИ ВАЛІДАЦІЇ")
    print("=" * 60)
    print(f"Консистентність передбачень: {'✅ PASS' if is_consistent else '❌ FAIL'}")
    print(f"Середня довіра: {avg_confidence:.4f}")
    print(f"Стандартне відхилення довіри: {std_confidence:.6f}")

    # Перевірка структури моделі
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n📈 Параметри моделі:")
    print(f"  Всього параметрів: {total_params:,}")
    print(f"  Тренованих параметрів: {trainable_params:,}")

    # Результати
    validation_results = {
        "model_path": model_path,
        "model_size_mb": os.path.getsize(model_path) / 1024 / 1024,
        "consistency_check": is_consistent,
        "predictions": [CLASSES[p] for p in predictions],
        "confidences": [float(c) for c in confidences],
        "avg_confidence": float(avg_confidence),
        "std_confidence": float(std_confidence),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "status": "PASS" if is_consistent else "FAIL"
    }

    # Збереження результатів
    with open("validation_report.json", "w") as f:
        json.dump(validation_results, f, indent=2)

    print("\n✅ Звіт збережено: validation_report.json")

    # Генерація Markdown звіту
    generate_markdown_report(validation_results)

    # Повернення exit code
    return 0 if is_consistent else 1


def create_stable_test_sample():
    """
    Створює стабільний тестовий зразок (синтетичний)
    для перевірки консистентності моделі
    """
    # Створюємо синусоїдальний сигнал з фіксованим seed
    torch.manual_seed(42)
    sample_rate = 16000
    duration = 1.0  # 1 секунда
    frequency = 440.0  # A4 note

    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = torch.sin(2 * np.pi * frequency * t).unsqueeze(0)

    return waveform, sample_rate


def generate_markdown_report(results: dict):
    """Генерує Markdown звіт"""

    report = f"""# 🔍 Звіт валідації моделі

## Загальна інформація

- **Модель:** `{results['model_path']}`
- **Розмір:** {results['model_size_mb']:.2f} MB
- **Статус:** {'✅ PASS' if results['status'] == 'PASS' else '❌ FAIL'}

## Тест консистентності

Модель була протестована 5 разів на одному і тому ж стабільному зразку.

### Передбачення:
"""

    for i, (pred, conf) in enumerate(zip(results['predictions'], results['confidences']), 1):
        report += f"{i}. **{pred}** (довіра: {conf:.4f})\n"

    report += f"""
### Статистика:

- **Консистентність:** {'✅ Всі передбачення однакові' if results['consistency_check'] else '❌ Передбачення відрізняються'}
- **Середня довіра:** {results['avg_confidence']:.4f}
- **Стандартне відхилення:** {results['std_confidence']:.6f}

## Параметри моделі

- **Всього параметрів:** {results['total_parameters']:,}
- **Тренованих параметрів:** {results['trainable_parameters']:,}

---
*Згенеровано автоматично GitHub Actions pipeline*
"""

    with open("validation_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("✅ Markdown звіт збережено: validation_report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Валідація моделі")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Шлях до моделі (.pt)")

    args = parser.parse_args()

    exit_code = validate_model_consistency(args.model_path)
    exit(exit_code)
