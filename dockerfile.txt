FROM python:3.9-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Настройка переменных окружения для отладки
ENV PYTHONUNBUFFERED=1
ENV DEBUG=True
ENV PYTHONDONTWRITEBYTECODE=1

# Создание рабочей директории
WORKDIR /app

# Копирование requirements и установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода приложения
COPY app.py .

# Открытие порта (для документации)
EXPOSE 5000

# Запуск с увеличенным таймаутом
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 300
