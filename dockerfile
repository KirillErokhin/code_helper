# Базовый образ с поддержкой CUDA 12.2 и Ubuntu 22.04 для работы с GPU
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Обновление пакетов и установка Python 3.10 и pip
RUN apt-get update && apt-get install -y python3.10 python3-pip

# Копируем все файлы из текущей директории в директорию /app внутри контейнера
COPY . /app

# Устанавливаем рабочую директорию контейнера как /app
WORKDIR /app

# Устанавливаем зависимости из файла requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Порт 7860 для работы с Gradio или другим веб-сервисом
EXPOSE 7860

# Устанавливаем переменную окружения для Gradio, чтобы сервер был доступен извне
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Указываем команду для запуска Python-приложения (запуск файла main.py)
CMD ["python3", "-u", "main.py"]
