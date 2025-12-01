#!/usr/bin/env python3
"""
Вимірювання latency та інших метрик моделі
"""

import argparse
import json
import time
import csv
from pathlib import Path
import warnings

import torch
import numpy as np

from model_utils import SmallCNN, wav_to_melspec, load_model

warnings.filterwarnings("ignore", category=UserWarning)

# Класи моделі
CLASSES = [
    "yes", "no",
    "up", "down", "zero", "wow", "visual", "two", "tree", "three",
    "stop", "six", "sheila", "seven", "right", "one", "off", "nine", "marvin",
    "left", "learn", "house", "happy", "go", "four", "forward", "follow", "five",
    "eight", "dog", "cat", "bird", "bed", "backward"
]


def measure_comprehensive_metrics(model_path: str, num_iterations: int = 100):
    """
    Вимірює комплексні метрики моделі
    """
    print("=" * 60)
    print("⏱️  ВИМІРЮВАННЯ МЕТРИК ПРОДУКТИВНОСТІ")
    print("=" * 60)

    # Завантаження моделі
    device = torch.device("cpu")
    model = load_model(SmallCNN, model_path, len(CLASSES), device)
    model.eval()

    print(f"✅ Модель завантажена: {model_path}")
    print(f"🔄 Кількість ітерацій: {num_iterations}")

    # Створення тестового входу
    torch.manual_seed(42)
    sample_rate = 16000
    duration = 1.0
    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = torch.sin(2 * np.pi * 440.0 * t).unsqueeze(0)
    melspec = wav_to_melspec(waveform, sample_rate).unsqueeze(0).to(device)

    print(f"📊 Розмір входу: {melspec.shape}")

    # Warm-up
    print("\n🔥 Warm-up (10 ітерацій)...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(melspec)

    # Вимірювання latency
    print("\n⏱️  Вимірювання inference latency...")
    latencies = []

    with torch.no_grad():
        for i in range(num_iterations):
            start_time = time.perf_counter()
            output = model(melspec)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            if (i + 1) % 20 == 0:
                print(f"  Прогрес: {i + 1}/{num_iterations}")

    # Статистика
    latencies = np.array(latencies)

    metrics = {
        "model_path": model_path,
        "num_iterations": num_iterations,
        "device": str(device),
        "input_shape": list(melspec.shape),
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "median": float(np.median(latencies)),
            "std": float(np.std(latencies)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99))
        },
        "throughput": {
            "requests_per_second": float(1000 / np.mean(latencies))
        },
        "model_info": {
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
    }

    # Виведення результатів
    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТИ ВИМІРЮВАНЬ")
    print("=" * 60)
    print(f"\n⏱️  Latency (мілісекунди):")
    print(f"  • Середнє:     {metrics['latency_ms']['mean']:.3f} ms")
    print(f"  • Медіана:     {metrics['latency_ms']['median']:.3f} ms")
    print(f"  • Мін:         {metrics['latency_ms']['min']:.3f} ms")
    print(f"  • Макс:        {metrics['latency_ms']['max']:.3f} ms")
    print(f"  • Std:         {metrics['latency_ms']['std']:.3f} ms")
    print(f"  • P95:         {metrics['latency_ms']['p95']:.3f} ms")
    print(f"  • P99:         {metrics['latency_ms']['p99']:.3f} ms")

    print(f"\n🚀 Throughput:")
    print(f"  • Запитів/сек: {metrics['throughput']['requests_per_second']:.2f}")

    print(f"\n📈 Модель:")
    print(f"  • Параметрів:  {metrics['model_info']['total_parameters']:,}")

    return metrics, latencies


def save_metrics(metrics: dict, latencies: np.ndarray, output_prefix: str):
    """Зберігає метрики у JSON та CSV форматах"""

    # JSON
    json_path = f"{output_prefix}.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ JSON метрики збережено: {json_path}")

    # CSV з детальними latency
    csv_path = f"{output_prefix}.csv"
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "latency_ms"])
        for i, latency in enumerate(latencies, 1):
            writer.writerow([i, latency])
    print(f"✅ CSV дані збережено: {csv_path}")

    # CSV з агрегованими метриками
    summary_csv_path = f"{output_prefix}_summary.csv"
    with open(summary_csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value", "unit"])
        writer.writerow(["mean_latency", metrics["latency_ms"]["mean"], "ms"])
        writer.writerow(["median_latency", metrics["latency_ms"]["median"], "ms"])
        writer.writerow(["std_latency", metrics["latency_ms"]["std"], "ms"])
        writer.writerow(["min_latency", metrics["latency_ms"]["min"], "ms"])
        writer.writerow(["max_latency", metrics["latency_ms"]["max"], "ms"])
        writer.writerow(["p95_latency", metrics["latency_ms"]["p95"], "ms"])
        writer.writerow(["p99_latency", metrics["latency_ms"]["p99"], "ms"])
        writer.writerow(["throughput", metrics["throughput"]["requests_per_second"], "req/s"])
        writer.writerow(["total_parameters", metrics["model_info"]["total_parameters"], "count"])
    print(f"✅ Агреговані метрики збережено: {summary_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Вимірювання метрик моделі")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Шлях до моделі (.pt)")
    parser.add_argument("--output", type=str, default="latency_metrics",
                        help="Префікс для вихідних файлів")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Кількість ітерацій для вимірювання")

    args = parser.parse_args()

    metrics, latencies = measure_comprehensive_metrics(args.model_path, args.iterations)
    save_metrics(metrics, latencies, args.output)

    print("\n✅ Вимірювання завершено!")
