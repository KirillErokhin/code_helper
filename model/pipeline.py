from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import torch


model_id = "https://huggingface.co/Qwen/Qwen2-7B-Instruct"
model_path = 'model/qwen/'

# Конфигурация для 4-битной квантизации с использованием библиотеки BitsAndBytes
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Загрузка модели с 4-битной квантизацией
    bnb_4bit_use_double_quant=True,  # Использование двойной квантизации для большей точности
    bnb_4bit_quant_type="nf4",  # Тип квантизации: NF4, улучшенная версия 4-битной квантизации
    bnb_4bit_compute_dtype=torch.bfloat16,  # Тип данных для вычислений: bfloat16, для экономии памяти и производительности
)

try:
    model = AutoModelForCausalLM.from_pretrained(
        'model/qwen/',
        torch_dtype=torch.float16,
        device_map="auto", # Выбор CPU / GPU для инференса модели
        quantization_config=quantization_config,
        low_cpu_mem_usage=True # Оптимизация использования памяти
    )
    tokenizer = AutoTokenizer.from_pretrained('model/qwen/')
except:
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quantization_config,
        cache_dir='model/qwen/', # Директория для кэширования модели
        low_cpu_mem_usage=True 
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct", cache_dir='model/qwen/')