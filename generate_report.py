#!/usr/bin/env python3
"""
Генерація фінального звіту з усіх артефактів pipeline
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime


def load_json_safe(filepath: str):
    """Безпечне завантаження JSON"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Не вдалося завантажити {filepath}: {e}")
        return None


def generate_report(artifacts_dir: str, output_file: str):
    """Генерує комплексний звіт"""

    print("=" * 60)
    print("📝 ГЕНЕРАЦІЯ ФІНАЛЬНОГО ЗВІТУ")
    print("=" * 60)

    artifacts_path = Path(artifacts_dir)

    # Завантаження артефактів
    validation_data = None
    metrics_data = None

    # Шукаємо validation report
    validation_json = artifacts_path / "validation-report" / "validation_report.json"
    if validation_json.exists():
        validation_data = load_json_safe(str(validation_json))
        print(f"✅ Знайдено звіт валідації")

    # Шукаємо performance metrics
    metrics_json = artifacts_path / "performance-metrics" / "latency_metrics.json"
    if metrics_json.exists():
        metrics_data = load_json_safe(str(metrics_json))
        print(f"✅ Знайдено метрики продуктивності")

    # Генерація Markdown звіту
    report = f"""# 🤖 CI/CD Pipeline - Звіт виконання

**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📋 Огляд

Цей звіт містить результати автоматичного pipeline для тренування та деплою моделі Speech Commands Classification.

"""

    # Секція валідації
    if validation_data:
        status_emoji = "✅" if validation_data.get("status") == "PASS" else "❌"
        report += f"""## 🔍 Валідація моделі {status_emoji}

### Результати тесту консистентності

- **Статус:** {validation_data.get('status', 'N/A')}
- **Консистентність передбачень:** {'✅ Пройдено' if validation_data.get('consistency_check') else '❌ Не пройдено'}
- **Розмір моделі:** {validation_data.get('model_size_mb', 0):.2f} MB
- **Середня довіра:** {validation_data.get('avg_confidence', 0):.4f}
- **Стандартне відхилення довіри:** {validation_data.get('std_confidence', 0):.6f}

### Передбачення на тестовому зразку

"""
        for i, (pred, conf) in enumerate(zip(
            validation_data.get('predictions', []),
            validation_data.get('confidences', [])
        ), 1):
            report += f"{i}. **{pred}** - довіра: {conf:.4f}\n"

        report += f"""
### Параметри моделі

- **Всього параметрів:** {validation_data.get('total_parameters', 0):,}
- **Тренованих параметрів:** {validation_data.get('trainable_parameters', 0):,}

"""

    # Секція метрик продуктивності
    if metrics_data:
        latency = metrics_data.get('latency_ms', {})
        throughput = metrics_data.get('throughput', {})

        report += f"""## ⏱️ Метрики продуктивності

### Inference Latency

| Метрика | Значення |
|---------|----------|
| Середнє | {latency.get('mean', 0):.3f} ms |
| Медіана | {latency.get('median', 0):.3f} ms |
| Мінімум | {latency.get('min', 0):.3f} ms |
| Максимум | {latency.get('max', 0):.3f} ms |
| Std Dev | {latency.get('std', 0):.3f} ms |
| P95 | {latency.get('p95', 0):.3f} ms |
| P99 | {latency.get('p99', 0):.3f} ms |

### Throughput

- **Запитів за секунду:** {throughput.get('requests_per_second', 0):.2f} req/s

### Деталі вимірювання

- **Кількість ітерацій:** {metrics_data.get('num_iterations', 'N/A')}
- **Пристрій:** {metrics_data.get('device', 'N/A')}
- **Розмір входу:** {metrics_data.get('input_shape', 'N/A')}

"""

    # Секція артефактів
    report += """## 📦 Згенеровані артефакти

### Модель
- `model_state_dict.pt` - навчена модель (state dict)
- `model_scripted.pt` - TorchScript модель

### Метрики
- `latency_metrics.json` - детальні метрики продуктивності
- `latency_metrics.csv` - CSV з latency даними
- `validation_report.json` - результати валідації

### Docker образи
- Inference image: `ghcr.io/<repository>:latest`
- Доступно в GitHub Container Registry

"""

    # Секція висновків
    report += """## 🎯 Висновки

"""

    if validation_data and validation_data.get("status") == "PASS":
        report += "✅ Модель пройшла валідацію успішно\n"
    else:
        report += "⚠️ Модель не пройшла валідацію\n"

    if metrics_data and latency.get('mean', 0) < 100:
        report += "✅ Latency в допустимих межах (<100ms)\n"
    elif metrics_data:
        report += "⚠️ Latency перевищує рекомендовані значення\n"

    report += """
## 🚀 Наступні кроки

1. Перевірте артефакти в GitHub Actions
2. Завантажте Docker образ з GHCR
3. Розгорніть сервіс у production середовищі

---

*Автоматично згенеровано GitHub Actions CI/CD Pipeline*
"""

    # Збереження звіту
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n✅ Звіт збережено: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генерація фінального звіту")
    parser.add_argument("--artifacts-dir", type=str, required=True,
                        help="Директорія з артефактами")
    parser.add_argument("--output", type=str, default="report.md",
                        help="Вихідний файл звіту")

    args = parser.parse_args()

    generate_report(args.artifacts_dir, args.output)

