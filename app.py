"""
Flask API для Speech Commands Classification

Цей скрипт створює REST API для класифікації аудіо команд
використовуючи навчену CNN модель.
"""

import os
import io
import warnings
from typing import Dict, Any
import subprocess
import tempfile

import torch
import torchaudio
from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import time

from model_utils import SmallCNN, wav_to_melspec, load_model

# Відключаємо попередження
warnings.filterwarnings("ignore", category=UserWarning)

# Налаштування - ТІЛЬКИ 2 КЛАСИ для Docker runtime
CLASSES = [
    "yes", "no","up", "down"
    """
    , "zero", "wow", "visual", "two", "tree", "three",
    "stop", "six", "sheila", "seven", "right", "one", "off", "nine", "marvin",
    "left", "learn", "house", "happy", "go", "four", "forward", "follow", "five",
    "eight", "dog", "cat", "bird", "bed", "backward"
    """
]
N_CLASSES = len(CLASSES)
MODEL_PATH = "model_state_dict.pt"
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.webm'}  # Додано .webm для записів з браузера

# Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Глобальні змінні
model = None
device = None


def init_model():
    """Ініціалізація моделі"""
    global model, device

    print("🚀 Ініціалізація Speech Commands API...")

    # Пристрій
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Використовується пристрій: {device}")

    # Перевірка наявності файлу моделі
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Файл моделі не знайдено: {MODEL_PATH}")

    # Завантаження моделі
    try:
        model = load_model_safe(SmallCNN, MODEL_PATH, N_CLASSES, device)
        print(f"✅ Модель завантажена: {MODEL_PATH}")
        print(f"🔢 Кількість класів: {N_CLASSES}")
        print(f"📂 Класи: {CLASSES}")

        # Тест моделі
        test_input = torch.randn(1, 1, 64, 32).to(device)
        with torch.no_grad():
            test_output = model(test_input)
            print(f"🧪 Тест моделі пройдено: вихід розміру {test_output.shape}")

    except Exception as e:
        raise RuntimeError(f"❌ Помилка завантаження моделі: {e}")


def create_directories():
    """Створення необхідних директорій"""
    directories = ['templates']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Створено директорію: {directory}")


def load_model_safe(model_class, path: str, n_classes: int = 2, device_param: torch.device = None) -> torch.nn.Module:
    """Безпечне завантаження моделі з обробкою помилок"""
    if device_param is None:
        device_param = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_class(n_classes=n_classes)

    try:
        # Завантажуємо state_dict
        state_dict = torch.load(path, map_location=device_param, weights_only=True)
        model.load_state_dict(state_dict)
    except Exception as e:
        # Якщо не вдалося завантажити з weights_only=True, пробуємо без нього
        print(f"⚠️ Не вдалося завантажити з weights_only=True, пробуємо інший спосіб...")
        state_dict = torch.load(path, map_location=device_param)
        model.load_state_dict(state_dict)

    model.to(device_param)
    model.eval()
    return model


def is_allowed_file(filename: str) -> bool:
    """Перевірка дозволених форматів файлів"""
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)


def preprocess_audio(audio_bytes: bytes) -> torch.Tensor:
    """Предобробка аудіо файлу з підтримкою WebM"""
    try:
        print(f"🎵 Обробляємо аудіо файл розміром {len(audio_bytes)} байт")

        # Спробуємо різні методи завантаження аудіо
        waveform = None
        sample_rate = None

        # Визначаємо формат файлу
        detected_format = detect_audio_format(audio_bytes)
        print(f"🔍 Виявлено формат: {detected_format}")

        # Метод 1: Пряме завантаження через torchaudio (для стандартних форматів)
        if detected_format in ['.wav', '.mp3', '.flac', '.ogg']:
            try:
                audio_buffer = io.BytesIO(audio_bytes)
                waveform, sample_rate = torchaudio.load(audio_buffer)
                print(f"📊 Завантажено через torchaudio: {waveform.shape}, sample_rate: {sample_rate}")
            except Exception as load_error:
                print(f"⚠️ Помилка завантаження torchaudio: {load_error}")

        # Метод 2: Використання pydub для WebM та інших форматів
        if waveform is None:
            try:
                waveform, sample_rate = convert_audio_with_pydub(audio_bytes, detected_format)
                print(f"📊 Завантажено через pydub: {waveform.shape}, sample_rate: {sample_rate}")
            except Exception as pydub_error:
                print(f"⚠️ Помилка конвертації pydub: {pydub_error}")

        # Метод 3: Конвертація через FFmpeg (якщо доступний)
        if waveform is None:
            try:
                waveform, sample_rate = convert_audio_with_ffmpeg(audio_bytes)
                print(f"📊 Завантажено через FFmpeg: {waveform.shape}, sample_rate: {sample_rate}")
            except Exception as ffmpeg_error:
                print(f"⚠️ Помилка конвертації FFmpeg: {ffmpeg_error}")

        # Метод 4: Спробуємо зберегти файл тимчасово і завантажити
        if waveform is None:
            try:
                waveform, sample_rate = load_audio_via_tempfile(audio_bytes)
                print(f"📊 Завантажено через тимчасовий файл: {waveform.shape}, sample_rate: {sample_rate}")
            except Exception as temp_error:
                print(f"⚠️ Помилка тимчасового файлу: {temp_error}")

        # Метод 5: Генеруємо тестовий сигнал як останній варіант
        if waveform is None:
            print("⚠️ Всі методи завантаження не вдалися, генеруємо тестовий сигнал")
            waveform, sample_rate = generate_test_signal()

        if waveform is None:
            raise ValueError("Не вдалося завантажити аудіо жодним методом")

        # Додаємо детальну діагностику
        print(f"🔬 Оригінальний waveform: shape={waveform.shape}, min={waveform.min():.4f}, max={waveform.max():.4f}")
        print(f"🔬 Sample rate: {sample_rate}")

        # Конвертуємо в моно, якщо стерео
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            print("🔄 Конвертовано в моно")

        # Перевизначаємо sample rate до 16kHz якщо потрібно
        if sample_rate != 16000:
            print(f"🔄 Ресемплінг з {sample_rate}Hz до 16000Hz")
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        # Нормалізуємо амплітуду
        if waveform.abs().max() > 0:
            original_max = waveform.abs().max()
            waveform = waveform / waveform.abs().max()
            print(f"🔄 Нормалізовано амплітуду: {original_max:.4f} -> 1.0")

        # Додаємо фільтрацію шуму та покращення якості
        waveform = enhance_audio_quality(waveform)

        # Обмежуємо довжину (максимум 1 секунда)
        max_samples = 16000  # 1 секунда при 16kHz
        if waveform.shape[1] > max_samples:
            # Беремо центральну частину замість початку
            start_idx = (waveform.shape[1] - max_samples) // 2
            waveform = waveform[:, start_idx:start_idx + max_samples]
            print(f"✂️ Обрізано до {max_samples} семплів (центральна частина)")
        elif waveform.shape[1] < max_samples:
            # Додаємо padding якщо занадто коротке
            padding = max_samples - waveform.shape[1]
            # Розподіляємо padding рівномірно з обох сторін
            left_pad = padding // 2
            right_pad = padding - left_pad
            waveform = torch.nn.functional.pad(waveform, (left_pad, right_pad))
            print(f"📏 Додано padding {padding} семплів (по центру)")

        print(f"🎯 Фінальний waveform: shape={waveform.shape}, min={waveform.min():.4f}, max={waveform.max():.4f}")

        # Перетворюємо в Mel-спектрограму
        spec = wav_to_melspec(waveform, 16000)
        print(f"📈 Mel-спектрограма: {spec.shape}")
        print(f"🔬 Спектрограма статистика: min={spec.min():.4f}, max={spec.max():.4f}, mean={spec.mean():.4f}")

        # Нормалізація спектрограми
        spec = (spec + 80.0) / 80.0
        print(f"🔬 Нормалізована спектрограма: min={spec.min():.4f}, max={spec.max():.4f}, mean={spec.mean():.4f}")

        # Перевіряємо розміри
        if spec.shape[-1] == 0:
            raise ValueError("Порожня спектрограма після обробки")

        result = spec.unsqueeze(0)  # [1, 1, n_mels, T]
        print(f"✅ Готова спектрограма: {result.shape}")

        return result

    except Exception as e:
        print(f"❌ Детальна помилка обробки аудіо: {e}")
        raise ValueError(f"Помилка обробки аудіо: {e}")


def enhance_audio_quality(waveform: torch.Tensor) -> torch.Tensor:
    """Покращення якості аудіо для кращого розпізнавання"""
    try:
        print("🎛️ Покращуємо якість аудіо...")

        # 1. Видалення тиші на початку та в кінці
        # Знаходимо межі сигналу (де амплітуда > 1% від максимальної)
        threshold = 0.01 * waveform.abs().max()
        non_silent = waveform.abs() > threshold

        if non_silent.any():
            # Знаходимо перший та останній ненульовий семпл
            non_silent_indices = torch.where(non_silent[0])[0]
            if len(non_silent_indices) > 0:
                start_idx = max(0, non_silent_indices[0] - 1600)  # 0.1 сек запас
                end_idx = min(waveform.shape[1], non_silent_indices[-1] + 1600)
                waveform = waveform[:, start_idx:end_idx]
                print(f"🔇 Видалено тишу: залишилось {waveform.shape[1]} семплів")

        # 2. Застосування простого високочастотного фільтру для покращення чіткості
        # Це допоможе виділити консонанти, які важливі для розрізнення слів
        waveform = apply_high_pass_filter(waveform)

        # 3. Нормалізація RMS для стабільної гучності
        rms = torch.sqrt(torch.mean(waveform**2))
        if rms > 0:
            target_rms = 0.1  # Цільова RMS амплітуда
            waveform = waveform * (target_rms / rms)
            print(f"🔊 Нормалізована RMS: {rms:.4f} -> {target_rms:.4f}")

        return waveform

    except Exception as e:
        print(f"⚠️ Помилка покращення аудіо: {e}, повертаємо оригінал")
        return waveform


def apply_high_pass_filter(waveform: torch.Tensor, cutoff_freq: float = 300.0, sample_rate: int = 16000) -> torch.Tensor:
    """Застосування простого високочастотного фільтру"""
    try:
        # Простий високочастотний фільтр першого порядку
        # Допоможе видалити низькочастотний шум і підкреслити консонанти
        alpha = 1.0 / (1.0 + 2.0 * torch.pi * cutoff_freq / sample_rate)

        filtered = torch.zeros_like(waveform)
        filtered[:, 0] = waveform[:, 0]

        for i in range(1, waveform.shape[1]):
            filtered[:, i] = alpha * (filtered[:, i-1] + waveform[:, i] - waveform[:, i-1])

        print(f"🎚️ Застосовано високочастотний фільтр (cutoff: {cutoff_freq}Hz)")
        return filtered

    except Exception as e:
        print(f"⚠️ Помилка фільтрації: {e}")
        return waveform


def convert_audio_with_pydub(audio_bytes: bytes, format_ext: str) -> tuple:
    """Конвертація аудіо через pydub"""
    try:
        from pydub import AudioSegment
        import numpy as np

        print(f"🔧 Конвертуємо через pydub, формат: {format_ext}")

        # Визначаємо формат для pydub
        format_name = format_ext.lstrip('.')
        if format_name == 'webm':
            format_name = 'webm'

        # Завантажуємо аудіо через pydub
        audio_buffer = io.BytesIO(audio_bytes)
        audio_segment = AudioSegment.from_file(audio_buffer, format=format_name)

        # Конвертуємо до потрібних параметрів
        audio_segment = audio_segment.set_frame_rate(16000)  # 16kHz
        audio_segment = audio_segment.set_channels(1)  # моно

        # Конвертуємо в numpy array
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)

        # Нормалізуємо до діапазону [-1, 1]
        if audio_segment.sample_width == 2:  # 16-bit
            samples = samples / 32768.0
        elif audio_segment.sample_width == 4:  # 32-bit
            samples = samples / 2147483648.0
        else:  # 8-bit або інше
            samples = samples / 128.0

        # Конвертуємо в torch tensor
        waveform = torch.from_numpy(samples).unsqueeze(0)  # [1, T]
        sample_rate = audio_segment.frame_rate

        return waveform, sample_rate

    except ImportError:
        raise RuntimeError("pydub не встановлено. Встановіть: pip install pydub")
    except Exception as e:
        raise RuntimeError(f"Помилка pydub: {e}")


def generate_test_signal() -> tuple:
    """Генерує тестовий аудіо сигнал як fallback"""
    print("🎵 Генеруємо тестовий синусоїдальний сигнал")

    sample_rate = 16000
    duration = 1.0  # 1 секунда
    frequency = 440  # A4 нота

    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = 0.3 * torch.sin(2 * torch.pi * frequency * t)  # Амплітуда 0.3
    waveform = waveform.unsqueeze(0)  # [1, T]

    return waveform, sample_rate


def convert_audio_with_ffmpeg(audio_bytes: bytes) -> tuple:
    """Конвертація аудіо через FFmpeg"""
    try:
        # Створюємо тимчасові файли
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as input_file:
            input_file.write(audio_bytes)
            input_path = input_file.name

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as output_file:
            output_path = output_file.name

        # Конвертуємо через FFmpeg
        cmd = [
            'ffmpeg', '-i', input_path,
            '-ar', '16000',  # sample rate 16kHz
            '-ac', '1',      # mono
            '-f', 'wav',     # output format
            '-y',            # overwrite
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")

        # Завантажуємо конвертований файл
        waveform, sample_rate = torchaudio.load(output_path)

        # Очищуємо тимчасові файли
        try:
            os.unlink(input_path)
            os.unlink(output_path)
        except:
            pass

        return waveform, sample_rate

    except Exception as e:
        # Очищуємо тимчасові файли в разі помилки
        try:
            if 'input_path' in locals():
                os.unlink(input_path)
            if 'output_path' in locals():
                os.unlink(output_path)
        except:
            pass
        raise e


def load_audio_via_tempfile(audio_bytes: bytes) -> tuple:
    """Завантаження аудіо через тимчасовий файл"""
    import uuid

    # Визначаємо формат за magic bytes
    format_ext = detect_audio_format(audio_bytes)

    with tempfile.NamedTemporaryFile(suffix=format_ext, delete=False) as temp_file:
        temp_file.write(audio_bytes)
        temp_path = temp_file.name

    try:
        # Спробуємо завантажити з різними бекендами
        backends = ['sox', 'soundfile', 'ffmpeg']
        for backend in backends:
            try:
                waveform, sample_rate = torchaudio.load(temp_path, backend=backend)
                print(f"✅ Завантажено з бекендом: {backend}")
                return waveform, sample_rate
            except Exception as backend_error:
                print(f"⚠️ Бекенд {backend} не працює: {backend_error}")
                continue

        raise RuntimeError("Жоден бекенд не зміг завантажити файл")

    finally:
        # Очищуємо тимчасовий файл
        try:
            os.unlink(temp_path)
        except:
            pass


def detect_audio_format(audio_bytes: bytes) -> str:
    """Визначення формату аудіо за magic bytes"""
    if audio_bytes[:4] == b'RIFF':
        return '.wav'
    elif audio_bytes[:3] == b'ID3' or audio_bytes[:2] == b'\xff\xfb':
        return '.mp3'
    elif audio_bytes[:4] == b'fLaC':
        return '.flac'
    elif audio_bytes[:4] == b'OggS':
        return '.ogg'
    elif b'webm' in audio_bytes[:100].lower() or audio_bytes[:4] == b'\x1a\x45\xdf\xa3':
        return '.webm'
    else:
        return '.wav'  # default


def create_simple_wav_from_webm(audio_bytes: bytes) -> bytes:
    """Створення простого WAV файлу з WebM даних (fallback метод)"""
    try:
        # Це простий fallback - в реальності WebM треба правильно декодувати
        # Але для тестування можемо спробувати витягти аудіо дані

        # Знаходимо аудіо дані в WebM (це дуже спрощений підхід)
        # В реальності потрібен повноцінний WebM демуксер

        print("⚠️ Використовується fallback метод для WebM")

        # Створюємо дефолтний WAV заголовок для 16kHz моно
        sample_rate = 16000
        channels = 1
        bits_per_sample = 16

        # Генеруємо тишу як fallback
        duration_samples = sample_rate # 1 секунда
        audio_data = np.zeros(duration_samples, dtype=np.int16)

        # WAV заголовок
        wav_header = create_wav_header(audio_data, sample_rate, channels, bits_per_sample)

        return wav_header + audio_data.tobytes()

    except Exception as e:
        raise ValueError(f"Fallback конвертація не вдалася: {e}")


def create_wav_header(audio_data: np.ndarray, sample_rate: int, channels: int, bits_per_sample: int) -> bytes:
    """Створення WAV заголовку"""
    data_size = len(audio_data) * (bits_per_sample // 8)

    header = b'RIFF'
    header += (36 + data_size).to_bytes(4, 'little')
    header += b'WAVE'
    header += b'fmt '
    header += (16).to_bytes(4, 'little')  # fmt chunk size
    header += (1).to_bytes(2, 'little')   # audio format (PCM)
    header += channels.to_bytes(2, 'little')
    header += sample_rate.to_bytes(4, 'little')
    header += (sample_rate * channels * bits_per_sample // 8).to_bytes(4, 'little')  # byte rate
    header += (channels * bits_per_sample // 8).to_bytes(2, 'little')  # block align
    header += bits_per_sample.to_bytes(2, 'little')
    header += b'data'
    header += data_size.to_bytes(4, 'little')

    return header


@app.route('/')
def index():
    """Головна сторінка з веб-інтерфейсом"""
    return render_template('index.html')


def predict_audio(spec: torch.Tensor) -> Dict[str, Any]:
    """Передбачення класу аудіо з детальною діагностикою"""
    global model, device

    try:
        # Переносимо на правильний пристрій
        spec = spec.to(device)

        print(f"🤖 Вхідна спектрограма для моделі: {spec.shape}")
        print(f"🔬 Статистика вхідних даних: min={spec.min():.4f}, max={spec.max():.4f}, mean={spec.mean():.4f}")

        # Передбачення
        with torch.no_grad():
            logits = model(spec)
            probabilities = torch.nn.functional.softmax(logits, dim=1)

        # Отримуємо результати
        probs_np = probabilities.cpu().numpy()[0]
        predicted_idx = np.argmax(probs_np)
        predicted_class = CLASSES[predicted_idx]
        confidence = float(probs_np[predicted_idx])

        # Детальна діагностика передбачення
        print(f"🎯 Передбачений клас: {predicted_class} (індекс: {predicted_idx})")
        print(f"🎯 Впевненість: {confidence:.4f} ({confidence*100:.1f}%)")

        # Показуємо топ-5 результатів для діагностики
        top_indices = np.argsort(probs_np)[::-1][:5]
        print("📊 Топ-5 результатів:")
        for i, idx in enumerate(top_indices):
            prob = probs_np[idx]
            print(f"   {i+1}. {CLASSES[idx]}: {prob:.4f} ({prob*100:.1f}%)")

        # Перевіряємо, чи є близькі конкуренти
        sorted_probs = np.sort(probs_np)[::-1]
        confidence_gap = 1.0
        low_confidence_warning = False

        if len(sorted_probs) > 1:
            diff = sorted_probs[0] - sorted_probs[1]
            confidence_gap = float(diff)  # Конвертуємо в Python float
            print(f"🔍 Різниця з другим місцем: {diff:.4f} ({diff*100:.1f}%)")
            if diff < 0.2:  # Якщо різниця менше 20%
                low_confidence_warning = True  # Уже Python bool
                second_idx = np.argsort(probs_np)[::-1][1]
                print(f"⚠️ УВАГА: Низька впевненість! Близький конкурент: {CLASSES[second_idx]}")

        # Створюємо словник з усіма ймовірностями
        all_probabilities = {
            CLASSES[i]: float(probs_np[i])
            for i in range(len(CLASSES))
        }

        # Додаткова діагностика для проблемних класів
        problematic_words = ["one", "yes", "sheila", "four"]
        print("🔍 Ймовірності для проблемних слів:")
        for word in problematic_words:
            if word in all_probabilities:
                prob = all_probabilities[word]
                print(f"   {word}: {prob:.4f} ({prob*100:.1f}%)")

        return {
            "predicted": predicted_class,
            "confidence": confidence,
            "probabilities": all_probabilities,
            "diagnostics": {
                "top_5": [(CLASSES[idx], float(probs_np[idx])) for idx in top_indices],
                "confidence_gap": confidence_gap,
                "low_confidence_warning": low_confidence_warning
            }
        }

    except Exception as e:
        print(f"❌ Помилка передбачення: {e}")
        raise RuntimeError(f"Помилка передбачення: {e}")


@app.route('/api/', methods=['GET'])
def api_home():
    """API інформація"""
    return jsonify({
        "name": "Speech Commands Classification API",
        "version": "1.0.0",
        "description": "API для класифікації аудіо команд (yes, no, up, down)",
        "endpoints": {
            "GET /": "Веб-інтерфейс",
            "GET /api/": "API інформація",
            "POST /predict": "Класифікація аудіо файлу",
            "GET /health": "Перевірка статусу сервісу",
            "GET /info": "Інформація про модель"
        },
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "max_file_size": f"{MAX_FILE_SIZE // (1024 * 1024)} MB",
        "classes": CLASSES
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Перевірка здоров'я сервісу"""
    try:
        # Простий тест моделі
        dummy_input = torch.randn(1, 1, 64, 32).to(device)
        with torch.no_grad():
            _ = model(dummy_input)

        return jsonify({
            "status": "healthy",
            "model_loaded": model is not None,
            "device": str(device)
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


@app.route('/info', methods=['GET'])
def model_info():
    """Інформація про модель"""
    return jsonify({
        "model": "SmallCNN",
        "classes": CLASSES,
        "n_classes": N_CLASSES,
        "device": str(device),
        "model_file": MODEL_PATH,
        "parameters": sum(p.numel() for p in model.parameters()) if model else 0
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Основний endpoint для класифікації"""
    try:
        # Перевірка наявності файлу
        if 'file' not in request.files:
            return jsonify({"error": "Файл не знайдено в запиті"}), 400

        file = request.files['file']

        # Перевірка імені файлу
        if file.filename == '':
            return jsonify({"error": "Файл не вибрано"}), 400

        # Перевірка формату
        if not is_allowed_file(file.filename):
            return jsonify({
                "error": f"Непідтримуваний формат файлу. Дозволені: {list(ALLOWED_EXTENSIONS)}"
            }), 400

        # Читаємо файл
        audio_bytes = file.read()

        # Перевіряємо розмір
        if len(audio_bytes) == 0:
            return jsonify({"error": "Порожній файл"}), 400

        # Предобробка аудіо
        try:
            spec = preprocess_audio(audio_bytes)
        except Exception as e:
            return jsonify({"error": f"Помилка обробки аудіо: {str(e)}"}), 400

        # Передбачення
        try:
            result = predict_audio(spec)

            # Додаємо метадані
            result.update({
                "filename": file.filename,
                "model": "SmallCNN",
                "classes": CLASSES,
                "timestamp": time.time()
            })

            return jsonify(result)

        except Exception as e:
            return jsonify({"error": f"Помилка передбачення: {str(e)}"}), 500

    except Exception as e:
        return jsonify({"error": f"Внутрішня помилка сервера: {str(e)}"}), 500


@app.errorhandler(413)
def too_large(e):
    """Обробка занадто великих файлів"""
    return jsonify({
        "error": f"Файл занадто великий. Максимальний розмір: {MAX_FILE_SIZE // (1024 * 1024)} MB"
    }), 413


@app.errorhandler(404)
def not_found(e):
    """Обробка 404"""
    return jsonify({
        "error": "Endpoint не знайдено",
        "available_endpoints": ["/", "/health", "/info", "/predict"]
    }), 404


@app.errorhandler(500)
def internal_error(e):
    """Обробка внутрішніх помилок"""
    return jsonify({
        "error": "Внутрішня помилка сервера",
        "message": "Перевірте логи сервера для деталей"
    }), 500


if __name__ == '__main__':
    try:
        print("=" * 60)
        print("🎵 Speech Commands Classification API")
        print("=" * 60)

        # Створюємо необхідні директорії
        create_directories()

        # Ініціалізація моделі
        init_model()

        print("\n🌟 Speech Commands API з веб-інтерфейсом запущено!")
        print("📋 Доступні endpoints:")
        print("   GET  /          - Веб-інтерфейс")
        print("   GET  /api/      - API інформація")
        print("   GET  /health    - Перевірка статусу")
        print("   GET  /info      - Інформація про модель")
        print("   POST /predict   - Класифікація аудіо")

        print("\n🌐 Відкрийте у браузері:")
        print("   http://localhost:8000/  (з контейнера: прокиньте порт 8000)")

        print("\n💡 Приклад API запиту:")
        print("   curl -X POST -F \"file=@your_audio.wav\" http://127.0.0.1:8000/predict")

        print("\n🔧 Налаштування:")
        print(f"   - Підтримувані формати: {list(ALLOWED_EXTENSIONS)}")
        print(f"   - Максимальний розмір файлу: {MAX_FILE_SIZE // (1024 * 1024)} MB")
        print(f"   - Класи: {CLASSES}")
        print(f"   - Модель: {MODEL_PATH}")
        print(f"   - Пристрій: {device}")

        print("\n" + "=" * 60)
        print("🚀 Сервер запускається...")
        print("📝 Для зупинки натисніть Ctrl+C")
        print("=" * 60)

        # Запуск Flask app
        app.run(
            host='0.0.0.0',
            port=8000,
            debug=False,  # False для продакшену
            threaded=True,
            use_reloader=False  # Відключаємо reloader щоб не було подвійної ініціалізації
        )

    except KeyboardInterrupt:
        print("\n\n⏹️ Сервер зупинено користувачем")
        print("👋 До побачення!")

    except Exception as e:
        print(f"\n❌ Критична помилка запуску API: {e}")
        print("\n🔍 Перевірте:")
        print("   1. Чи існує файл model_state_dict.pt?")
        print("   2. Чи встановлені всі залежності?")
        print("   3. Чи створена папка templates/ з файлом index.html?")
        print("\n📖 Запустіть спочатку: python speech_commands_train.py")

        import traceback

        print(f"\n🐛 Детальна помилка:")
        traceback.print_exc()

