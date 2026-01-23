import sys
import os
import logging
from src.config import BASE_PATH 

def init_logger():
    # Определение формата логов с указанием пути и номера строки
    log_format = "%(asctime)s | %(name)s - %(levelname)s - %(message)s | %(pathname)s:%(lineno)d"
    
    # Создание объекта логгера
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)  # Установка уровня логирования по умолчанию
    
    # Создание обработчика для логирования в stderr
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(stream_handler)
    
    # Создание обработчика для логирования в файл
    log_file_path = os.path.join(BASE_PATH, "logs", "system.log")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)  # Убедитесь, что каталог для логов существует
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")  # Указание кодировки
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    return logger

# Инициализация логгера
logger = init_logger()