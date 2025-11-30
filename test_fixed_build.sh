#!/bin/bash

# ============================================================================
# Швидкий тест виправленої Docker збірки
# ============================================================================

set -e

echo "════════════════════════════════════════════════════════════════"
echo "🔧 Тестування виправленої Docker збірки"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Кольори
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

IMAGE_NAME="speech-commands-app"
CONTAINER_NAME="speech-app"

echo -e "${BLUE}📋 Виправлення які були зроблені:${NC}"
echo "  ✅ Додано soundfile backend для torchaudio"
echo "  ✅ Зменшено кількість класів з 34 до 4"
echo "  ✅ Зменшено BATCH_SIZE з 64 до 32"
echo "  ✅ Зменшено EPOCHS з 10 до 3"
echo "  ✅ Встановлено TORCHAUDIO_BACKEND=soundfile"
echo ""

echo -e "${YELLOW}🧹 Очищення попередніх контейнерів...${NC}"
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true
echo -e "${GREEN}✓ Очищено${NC}"
echo ""

echo -e "${YELLOW}🏗️  Збірка Docker образу...${NC}"
echo "⏰ Очікуваний час: 5-10 хвилин"
echo "📊 Буде навчено модель на ~2,400 зразках (4 класи)"
echo ""

START_TIME=$(date +%s)

if docker build -t $IMAGE_NAME . 2>&1 | tee /tmp/docker_build.log; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo ""
    echo -e "${GREEN}✓ Збірка завершена успішно!${NC}"
    echo "⏱️  Час збірки: $DURATION секунд ($((DURATION / 60)) хвилин)"
else
    echo ""
    echo -e "${RED}✗ Помилка збірки!${NC}"
    echo "📋 Перегляньте логи в /tmp/docker_build.log"
    exit 1
fi

echo ""
echo -e "${YELLOW}📊 Інформація про образ...${NC}"
docker images $IMAGE_NAME --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
echo ""

echo -e "${YELLOW}🔍 Перевірка моделі в образі...${NC}"
if docker run --rm $IMAGE_NAME ls -lh /app/model_state_dict.pt; then
    echo -e "${GREEN}✓ Модель знайдена в образі${NC}"
else
    echo -e "${RED}✗ Модель відсутня!${NC}"
    exit 1
fi
echo ""

echo -e "${YELLOW}🚀 Запуск контейнера...${NC}"
if docker run -d -p 8000:8000 --name $CONTAINER_NAME $IMAGE_NAME; then
    echo -e "${GREEN}✓ Контейнер запущено${NC}"
else
    echo -e "${RED}✗ Помилка запуску${NC}"
    exit 1
fi
echo ""

echo -e "${YELLOW}⏳ Очікування ініціалізації (10 секунд)...${NC}"
for i in {10..1}; do
    echo -n "$i... "
    sleep 1
done
echo ""
echo ""

echo -e "${YELLOW}📋 Логи контейнера:${NC}"
docker logs --tail 30 $CONTAINER_NAME
echo ""

echo -e "${YELLOW}🏥 Health check...${NC}"
MAX_RETRIES=15
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if HEALTH_RESPONSE=$(curl -f -s http://localhost:8000/health 2>/dev/null); then
        echo -e "${GREEN}✓ Health check пройдено${NC}"
        echo "📊 Відповідь:"
        echo "$HEALTH_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$HEALTH_RESPONSE"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
            echo -e "${RED}✗ Health check не пройдено після $MAX_RETRIES спроб${NC}"
            echo ""
            echo "📋 Логи помилок:"
            docker logs $CONTAINER_NAME
            exit 1
        fi
        echo -n "."
        sleep 2
    fi
done
echo ""

echo -e "${YELLOW}ℹ️  Інформація про модель...${NC}"
if INFO_RESPONSE=$(curl -s http://localhost:8000/info); then
    echo "$INFO_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$INFO_RESPONSE"
    echo -e "${GREEN}✓ API працює коректно${NC}"
else
    echo -e "${RED}✗ Помилка отримання інформації${NC}"
fi
echo ""

echo "════════════════════════════════════════════════════════════════"
echo -e "${GREEN}🎉 Тестування завершено успішно!${NC}"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo -e "${BLUE}📊 Підсумок:${NC}"
echo "  ✅ Збірка: Успішно ($DURATION секунд)"
echo "  ✅ Модель: Навчена та збережена"
echo "  ✅ Контейнер: Запущено"
echo "  ✅ API: Працює"
echo "  ✅ Класи: yes, no, up, down (4 класи)"
echo ""
echo -e "${BLUE}🌐 Доступ до сервісу:${NC}"
echo "  Web UI:    http://localhost:8000"
echo "  Health:    http://localhost:8000/health"
echo "  Info:      http://localhost:8000/info"
echo "  API Docs:  http://localhost:8000/api/"
echo ""
echo -e "${BLUE}📝 Корисні команди:${NC}"
echo "  Логи:              docker logs -f $CONTAINER_NAME"
echo "  Зупинити:          docker stop $CONTAINER_NAME"
echo "  Перезапустити:     docker restart $CONTAINER_NAME"
echo "  Видалити:          docker rm -f $CONTAINER_NAME"
echo "  Видалити образ:    docker rmi $IMAGE_NAME"
echo ""
echo -e "${BLUE}🧪 Тестування API:${NC}"
echo "  curl http://localhost:8000/health"
echo "  curl http://localhost:8000/info"
echo "  curl -X POST -F 'file=@audio.wav' http://localhost:8000/predict"
echo ""
echo "════════════════════════════════════════════════════════════════"
