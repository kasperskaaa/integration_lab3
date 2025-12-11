import torch
import torch.nn as nn
import torchaudio
import time
import os
from typing import Tuple

# Константи
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512


class SmallCNN(nn.Module):
    """Полегшена CNN модель для класифікації аудіо команд (оптимізована для Docker)"""

    def __init__(self, n_classes: int = 2):
        super(SmallCNN, self).__init__()

        self.features = nn.Sequential(
            # Перший блок - 8 фільтрів (замість 16)
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Другий блок - 16 фільтрів (замість 32)
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Третій блок - 32 фільтри (замість 64)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((8, 16))  # Зменшено розмір (8×16 замість 16×32)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 16, 128),  # Зменшено до 128 нейронів (замість 256)
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def wav_to_melspec(waveform: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Перетворює аудіо хвилю в Mel-спектрограму"""
    # Resample якщо потрібно
    if sample_rate != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    # Mel-спектрограма
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    # Перетворення в dB
    mel_spec = mel_transform(waveform)
    mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)

    return mel_spec_db


def measure_latency(model: nn.Module, example_input: torch.Tensor, runs: int = 30) -> float:
    """Вимірює середню латентність інференсу моделі"""
    model.eval()

    # Warm-up
    with torch.no_grad():
        for _ in range(3):
            _ = model(example_input)

    # Вимірювання
    times = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.time()
            _ = model(example_input)
            end = time.time()
            times.append((end - start) * 1000)  # в мілісекундах

    return sum(times) / len(times)


def save_model(model: nn.Module, path: str, n_classes: int = None, class_names: list = None) -> int:
    """Зберігає модель (state_dict) та метадані, повертає розмір файлу"""
    # Створюємо об'єкт для збереження
    save_dict = {
        'state_dict': model.state_dict(),
        'n_classes': n_classes if n_classes is not None else model.classifier[-1].out_features,
    }

    # Додаємо назви класів якщо передані
    if class_names is not None:
        save_dict['class_names'] = class_names

    torch.save(save_dict, path)
    return os.path.getsize(path)


def export_torchscript(model: nn.Module, example_input: torch.Tensor, path: str) -> int:
    """Експортує модель в TorchScript та повертає розмір файлу"""
    model.eval()
    scripted_model = torch.jit.trace(model, example_input)
    torch.jit.save(scripted_model, path)
    return os.path.getsize(path)


def load_model(model_class, path: str, n_classes: int = None, device: torch.device = None) -> nn.Module:
    """Завантажує модель з збереженого state_dict та метаданих"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Завантажуємо збережені дані
    checkpoint = torch.load(path, map_location=device)

    # Перевіряємо формат (старий або новий)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # Новий формат з метаданими
        state_dict = checkpoint['state_dict']
        saved_n_classes = checkpoint.get('n_classes', n_classes)

        # Використовуємо збережену кількість класів якщо не передана явно
        if n_classes is None:
            n_classes = saved_n_classes
    else:
        # Старий формат - тільки state_dict
        state_dict = checkpoint
        if n_classes is None:
            raise ValueError("n_classes must be provided for old format models")

    # Створюємо модель з правильною кількістю класів
    model = model_class(n_classes=n_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_model_metadata(path: str) -> dict:
    """Отримує метадані моделі без завантаження в пам'ять"""
    checkpoint = torch.load(path, map_location='cpu')

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        return {
            'n_classes': checkpoint.get('n_classes'),
            'class_names': checkpoint.get('class_names'),
            'has_metadata': True
        }
    else:
        # Старий формат - намагаємось визначити з розміру останнього шару
        if 'classifier.4.weight' in checkpoint:
            n_classes = checkpoint['classifier.4.weight'].shape[0]
            return {
                'n_classes': n_classes,
                'class_names': None,
                'has_metadata': False
            }
        return {
            'n_classes': None,
            'class_names': None,
            'has_metadata': False
        }
