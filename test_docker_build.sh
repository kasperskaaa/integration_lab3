#!/bin/bash

# Скрипт для тестування багатоступеневої Docker збірки
# Speech Commands Classification Project

set -e

echo "════════════════════════════════════════════════════════════════"
echo "🐳 Тестування багатоступеневої Docker збірки"
echo "════════════════════════════════════════════════════════════════"

# Кольори для виводу
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Назва образу і контейнера
IMAGE_NAME="speech-commands-app"
CONTAINER_NAME="speech-app-test"

# Функція для виводу кроків
step() {
    echo -e "\n${YELLOW}► $1${NC}"
}

# Функція для успіху
success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Функція для помилки
error() {
    echo -e "${RED}✗ $1${NC}"
    exit 1
}

# Перевірка чи запущений Docker
step "Перевірка Docker..."
if ! docker info > /dev/null 2>&1; then
    error "Docker не запущений або не встановлений!"
fi
success "Docker працює"

# Очищення попередніх контейнерів та образів (опціонально)
step "Очищення попередніх запусків..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true
success "Очищено"

# Збірка образу
step "Збірка Docker образу..."
echo "⏰ Це може зайняти кілька хвилин (особливо якщо модель треба навчити)..."

if docker build -t $IMAGE_NAME .; then
    success "Образ успішно збудовано"
else
    error "Помилка збірки образу"
fi

# Перевірка розміру образу
step "Інформація про образ..."
IMAGE_SIZE=$(docker images $IMAGE_NAME --format "{{.Size}}" | head -1)
echo "📦 Розмір образу: $IMAGE_SIZE"

# Перевірка наявності моделі в образі
step "Перевірка наявності моделі в образі..."
if docker run --rm $IMAGE_NAME ls -lh /app/model_state_dict.pt 2>/dev/null; then
    success "Модель знайдена в образі"
else
    error "Модель відсутня в образі!"
fi

# Запуск контейнера
step "Запуск контейнера..."
if docker run -d -p 8000:8000 --name $CONTAINER_NAME $IMAGE_NAME; then
    success "Контейнер запущено"
else
    error "Помилка запуску контейнера"
fi

# Очікування старту (5 секунд)
step "Очікування ініціалізації сервісу..."
for i in {5..1}; do
    echo -n "$i... "
    sleep 1
done
echo ""

# Перевірка логів
step "Перевірка логів контейнера..."
docker logs $CONTAINER_NAME | tail -20

# Health check
step "Перевірка health endpoint..."
MAX_RETRIES=10
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
        success "Health check пройдено"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
            error "Health check не пройдено після $MAX_RETRIES спроб"
        fi
        echo -n "."
        sleep 2
    fi
done

# Отримання інформації про модель
step "Інформація про модель..."
if curl -s http://localhost:8000/info | python3 -m json.tool; then
    success "API працює коректно"
else
    error "Помилка отримання інформації про модель"
fi

# Виведення корисної інформації
echo ""
echo "════════════════════════════════════════════════════════════════"
echo -e "${GREEN}✓ Тестування завершено успішно!${NC}"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📋 Корисні команди:"
echo ""
echo "  Переглянути логи:"
echo "    docker logs -f $CONTAINER_NAME"
echo ""
echo "  Відкрити веб-інтерфейс:"
echo "    open http://localhost:8000"
echo ""
echo "  Тестування API:"
echo "    curl -X POST -F \"file=@audio.wav\" http://localhost:8000/predict"
echo ""
echo "  Зупинити контейнер:"
echo "    docker stop $CONTAINER_NAME"
echo ""
echo "  Видалити контейнер:"
echo "    docker rm $CONTAINER_NAME"
echo ""
echo "  Видалити образ:"
echo "    docker rmi $IMAGE_NAME"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "🌐 API доступний на: http://localhost:8000"
echo "════════════════════════════════════════════════════════════════"
