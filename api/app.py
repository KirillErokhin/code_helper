from model.pipeline import model, tokenizer


prompt = "Ты помощник для работы с языками программирования. Обязательно пиши комментарии к коду."

def generate_answer(text):
    """
    Генерирует ответ на заданный текстовый запрос с использованием предварительно обученной модели языка.

    Параметры:
        text (str): Текстовый запрос для генерации ответа.

    Возвращает:
        str: Сгенерированный ответ.
    """

    # Создание шаблона чата с промтом и вводом пользователя
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": prompt}
    ]

    # Применение шаблона чата к вводу текста и генерация промта для модели
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Подготовка ввода текста для модели
    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

    # Генерация ответа с использованием модели
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )

    # Извлечение ответа из идентификаторов
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    # Декодирование ответа из идентификаторов в текст
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

