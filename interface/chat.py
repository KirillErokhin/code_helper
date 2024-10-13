from threading import Thread

from transformers import TextIteratorStreamer
import gradio as gr 

from model.pipeline import model, tokenizer


def chat_function_stream(message, history, system_prompt, max_new_tokens=512, temperature=0.7, top_p=0.8, top_k=20):
    """
    Генерирует поток ответов на заданный текстовый запрос с использованием предварительно обученной модели языка.

    Параметры:
        message (str): Текстовый запрос для генерации ответа.
        history (list): История предыдущих сообщений в чате.
        system_prompt (str): Промт для настройки модели на генерацию ответов.
        max_new_tokens (int, optional): Максимальное количество новых токенов для генерации (по умолчанию: 512).
        temperature (float, optional): Температура для генерации ответа (по умолчанию: 0.7).
        top_p (float, optional): Ограничение по вероятности для генерации ответа (по умолчанию: 0.8).
        top_k (int, optional): Ограничение по количеству наиболее вероятных токенов для генерации ответа (по умолчанию: 20).

    Возвращает:
        generator: Поток ответов в виде последовательности строк.
    """
    messages = [{"role": "system","content": system_prompt}]

    # Если история пуста, добавляем только текущее сообщение пользователя
    if len(history) == 0:
        messages.append({"role": "user", "content": message})
    else:
        # Если есть история, добавляем предыдущие сообщения и ответы
        for item in history:
            messages.append({"role": "user", "content": item[0]})  # сообщение пользователя
            messages.append({"role": "assistent", "content": item[1]})  # ответ ассистента
        # Добавляем текущее сообщение пользователя
        messages.append({"role": "user", "content": message})
        
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,)
    model_inputs = tokenizer([prompt], return_tensors="pt").to('cuda')

    # Создаем стример для поэтапной генерации ответа
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=None)

    # Настраиваем параметры генерации
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature
        )

    # Запускаем процесс генерации в отдельном потоке
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    # Частично собираем и передаем генерируемый ответ
    partial_message = ""
    for new_token in streamer:
        if new_token != '<': # Игнорируем спецсимволы
            partial_message += new_token
            yield partial_message # Возвращаем частично сгенерированное сообщение

# Создаем интерфейс для взаимодействия с моделью через Gradio
chat = gr.ChatInterface(
    chat_function_stream, # Функция для потоковой генерации
    textbox=gr.Textbox(placeholder="Enter message here", container=False, scale = 7),
    chatbot=gr.Chatbot(height=700),
    additional_inputs=[
        gr.Textbox("You are helpful AI", label="System Prompt"),
        gr.Slider(500,4000, label="Max New Tokens"),
        gr.Slider(0.001,1, label="Temperature"),
        gr.Slider(0,1, label="Top P"),
        gr.Slider(20,1000, label="Top K")
    ]
    )