from flask import Flask, request, jsonify
import pytesseract
from PIL import Image, UnidentifiedImageError
import io
import base64
from pdf2image import convert_from_bytes
import cv2
import numpy as np
import fitz  # PyMuPDF
import os
import hashlib
import logging
import datetime
import traceback
from werkzeug.utils import secure_filename

# Проверка наличия необходимых библиотек
try:
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Qdrant
    from langchain.schema import Document
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False
    print("ВНИМАНИЕ: Библиотеки для векторного поиска не установлены. Функционал сохранения в Qdrant недоступен.")

# Настройка логирования
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Константы и конфигурация
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'}
OCR_LANGUAGES = os.environ.get("OCR_LANGUAGES", "rus+eng")
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "dm_docs")
VECTOR_SIZE = 1536

# Максимальный размер загружаемого файла (20 МБ)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024

# Проверка наличия обязательных переменных окружения
required_vars = []
if VECTOR_SEARCH_AVAILABLE:
    required_vars = ["OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]
    
missing_vars = [var for var in required_vars if not os.environ.get(var)]
if missing_vars:
    logger.warning(f"Отсутствуют следующие переменные окружения: {', '.join(missing_vars)}. "
                  f"Некоторые функции могут быть недоступны.")

def allowed_file(filename):
    """Проверка допустимого расширения файла"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def ping():
    """Эндпоинт для проверки работоспособности сервера"""
    logger.info("Получен запрос к корневому эндпоинту")
    return "👋 OCR-сервис работает! Version 1.0.1"

@app.route("/health")
def health():
    """Эндпоинт для проверки здоровья сервиса"""
    # Проверяем все необходимые сервисы
    health_status = {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "services": {
            "tesseract": "ok",
            "opencv": "ok",
            "vector_search": "disabled" if not VECTOR_SEARCH_AVAILABLE else "ok"
        },
        "environment": {
            "ocr_languages": OCR_LANGUAGES,
            "max_file_size_mb": app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
        }
    }
    
    # Проверяем переменные окружения
    if VECTOR_SEARCH_AVAILABLE:
        for var in required_vars:
            health_status["services"][var.lower()] = "ok" if os.environ.get(var) else "missing"
    
    # Если что-то не в порядке, меняем общий статус
    if any(value != "ok" for service, value in health_status["services"].items() 
           if service not in ["vector_search"]):
        health_status["status"] = "degraded"
    
    return jsonify(health_status)

@app.route("/test_ocr", methods=["POST"])
def test_ocr():
    """Упрощенный эндпоинт для тестирования OCR"""
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "Файл не был загружен"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "Выбран пустой файл"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            "status": "error", 
            "message": f"Недопустимый тип файла. Разрешены только: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400
    
    try:
        image = Image.open(file.stream)
        text = pytesseract.image_to_string(image, lang=OCR_LANGUAGES)
        
        return jsonify({
            "status": "ok",
            "text": text,
            "languages": OCR_LANGUAGES
        })
    except Exception as e:
        logger.error(f"Ошибка при тестировании OCR: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Ошибка при обработке изображения: {str(e)}"
        }), 500

def preprocess_image(image, preprocessing_level="default"):
    """
    Предобработка изображения с различными уровнями обработки
    
    :param image: PIL Image объект
    :param preprocessing_level: Уровень обработки (default, light, aggressive)
    :return: Обработанное изображение как numpy array
    """
    try:
        img = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        if preprocessing_level == "light":
            # Легкая обработка
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            return blur
        
        elif preprocessing_level == "aggressive":
            # Агрессивная обработка для сложных документов
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            # Адаптивная бинаризация
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
            # Морфологические операции для удаления шума
            kernel = np.ones((1, 1), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            return opening
        
        else:  # default
            # Стандартная обработка
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
            return thresh
    except Exception as e:
        logger.error(f"Ошибка при предобработке изображения: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def extract_images_from_pdf(pdf_bytes):
    """Извлечение изображений из PDF файла"""
    images = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num, page in enumerate(doc):
            # Извлекаем изображения из страницы
            for img_index, img in enumerate(page.get_images(full=True)):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    img_bytes = base_image["image"]
                    encoded = base64.b64encode(img_bytes).decode('utf-8')
                    images.append({
                        "image": encoded,
                        "page": page_num + 1,
                        "index": img_index
                    })
                except Exception as e:
                    logger.error(f"Ошибка при извлечении изображения {img_index} со страницы {page_num}: {str(e)}")
    except Exception as e:
        logger.error(f"Ошибка при обработке PDF документа: {str(e)}")
        logger.error(traceback.format_exc())
    
    return images

def init_qdrant():
    """Инициализация клиента Qdrant и коллекции"""
    if not VECTOR_SEARCH_AVAILABLE:
        raise ValueError("Функционал векторного поиска недоступен. Проверьте установку зависимостей.")
        
    # Получение переменных окружения
    openai_key = os.environ.get("OPENAI_API_KEY")
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_key = os.environ.get("QDRANT_API_KEY")
    
    if not openai_key:
        logger.error("OPENAI_API_KEY не установлен")
        raise ValueError("OPENAI_API_KEY не установлен")
    
    if not qdrant_url:
        logger.error("QDRANT_URL не установлен")
        raise ValueError("QDRANT_URL не установлен")
    
    if not qdrant_key:
        logger.error("QDRANT_API_KEY не установлен")
        raise ValueError("QDRANT_API_KEY не установлен")
    
    # Создание клиентов
    embed_fn = OpenAIEmbeddings(openai_api_key=openai_key)
    raw_client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    
    # Создание коллекции, если не существует
    if not raw_client.collection_exists(collection_name=COLLECTION_NAME):
        logger.info(f"Создание новой коллекции {COLLECTION_NAME}")
        raw_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    
    client = Qdrant(
        url=qdrant_url,
        api_key=qdrant_key,
        prefer_grpc=False,
        embedding_function=embed_fn,
    )
    
    return client

@app.route("/ocr", methods=["POST"])
def ocr():
    """Эндпоинт для OCR-обработки документов"""
    try:
        logger.info("Получен запрос к эндпоинту /ocr")
        return jsonify({"status": "test", "message": "Simplified response"}), 200
        
        # Проверка наличия файла
        if 'file' not in request.files:
            logger.warning("Файл не был загружен")
            return jsonify({"error": "Файл не был загружен"}), 400
        
        file = request.files['file']
        
        # Проверка пустого файла
        if file.filename == '':
            logger.warning("Выбран пустой файл")
            return jsonify({"error": "Выбран пустой файл"}), 400
        
        # Проверка допустимого расширения
        if not allowed_file(file.filename):
            logger.warning(f"Недопустимый тип файла: {file.filename}")
            return jsonify({
                "error": f"Недопустимый тип файла. Разрешены только: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
        
        # Получение параметров запроса
        preprocessing_level = request.form.get('preprocessing', 'default')
        ocr_options = request.form.get('ocr_options', 'normal')
        store_in_qdrant = request.form.get('store_in_qdrant', 'true').lower() == 'true'
        
        # Проверка, возможно ли сохранение в Qdrant
        if store_in_qdrant and not VECTOR_SEARCH_AVAILABLE:
            logger.warning("Запрошено сохранение в Qdrant, но функционал недоступен")
            store_in_qdrant = False
        
        # Чтение файла
        filename = secure_filename(file.filename.lower())
        file_bytes = file.read()
        
        logger.info(f"Обрабатываю файл: {filename}, размер: {len(file_bytes)/1024:.2f} КБ")
        
        results = []
        all_text = ""
        extracted_images = []
        
        # Обработка PDF
        if filename.endswith('.pdf'):
            try:
                logger.info("Начинаю обработку PDF")
                images = convert_from_bytes(file_bytes)
                logger.info(f"PDF содержит {len(images)} страниц")
                
                for page_num, img in enumerate(images):
                    logger.info(f"Обрабатываю страницу {page_num+1}/{len(images)}")
                    pre_img = preprocess_image(img, preprocessing_level)
                    
                    # Конфигурация OCR для различных опций
                    config = ""
                    if ocr_options == "accurate":
                        config = "--oem 1 --psm 6"
                    elif ocr_options == "fast":
                        config = "--oem 0 --psm 3"
                    
                    text = pytesseract.image_to_string(pre_img, lang=OCR_LANGUAGES, config=config)
                    results.append({"page": page_num + 1, "text": text})
                    all_text += f"\n{text}"
                
                # Извлечение изображений из PDF
                logger.info("Извлекаю изображения из PDF")
                extracted_images = extract_images_from_pdf(file_bytes)
                logger.info(f"Извлечено {len(extracted_images)} изображений")
                
            except Exception as e:
                logger.error(f"Ошибка при обработке PDF: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({"error": f"Ошибка при обработке PDF: {str(e)}"}), 500
        
        # Обработка изображения
        else:
            try:
                logger.info("Начинаю обработку изображения")
                image = Image.open(io.BytesIO(file_bytes))
                pre_img = preprocess_image(image, preprocessing_level)
                
                config = ""
                if ocr_options == "accurate":
                    config = "--oem 1 --psm 6"
                elif ocr_options == "fast":
                    config = "--oem 0 --psm 3"
                
                text = pytesseract.image_to_string(pre_img, lang=OCR_LANGUAGES, config=config)
                results.append({"page": 1, "text": text})
                all_text = text
                
                # Кодирование изображения в base64
                buffered = io.BytesIO()
                image.save(buffered, format=image.format if image.format else "JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                extracted_images = [{"image": img_str, "page": 1, "index": 0}]
                
            except UnidentifiedImageError:
                logger.error(f"Не удалось распознать формат изображения: {filename}")
                return jsonify({"error": "Не удалось распознать формат изображения"}), 400
            except Exception as e:
                logger.error(f"Ошибка при обработке изображения: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({"error": f"Ошибка при обработке изображения: {str(e)}"}), 500
        
        # Генерация хеша
        hash_id = hashlib.md5(all_text.encode('utf-8')).hexdigest()
        logger.info(f"Создан хеш документа: {hash_id}")
        
        # Сохранение в Qdrant, если параметр установлен
        if store_in_qdrant and VECTOR_SEARCH_AVAILABLE:
            try:
                logger.info("Начинаю сохранение в Qdrant")
                # Инициализация Qdrant
                client = init_qdrant()
                
                # Создание документов для сохранения
                documents = [
                    Document(
                        page_content=page["text"],
                        metadata={
                            "page": page["page"],
                            "source": filename,
                            "hash": hash_id,
                            "file_type": os.path.splitext(filename)[1][1:],
                            "timestamp": datetime.datetime.now().isoformat(),
                        },
                    )
                    for page in results
                ]
                
                # Добавление документов в коллекцию
                client.add_documents(documents, collection_name=COLLECTION_NAME)
                logger.info(f"Документ {hash_id} успешно сохранен в Qdrant")
                
            except Exception as e:
                logger.error(f"Ошибка при сохранении в Qdrant: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({
                    "status": "partial_success",
                    "hash": hash_id,
                    "pages": len(results),
                    "text": results,
                    "images": extracted_images,
                    "error": f"Текст успешно распознан, но произошла ошибка при сохранении в Qdrant: {str(e)}"
                })
        
        # Формирование успешного ответа
        logger.info("Успешно завершена обработка запроса")
        return jsonify({
            "status": "ok",
            "hash": hash_id,
            "pages": len(results),
            "text": results,
            "images": extracted_images
        })
        
    except Exception as e:
        logger.error(f"Общая ошибка при обработке запроса: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Произошла ошибка при обработке запроса: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("DEBUG", "False").lower() == "true"
    logger.info(f"Запуск сервера на порту {port}, режим отладки: {debug_mode}")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
