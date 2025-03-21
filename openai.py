import logging
import os
from typing import Optional

import requests


class OpenAI:
    """
    Класс для взаимодействия с OpenAI API.
    """

    def __init__(self, openai_api_key: str = None, openai_url: str = None):
        """
        Инициализирует класс OpenAI.
        :param openai_api_key: Ключ API OpenAI (если не задан, берется из переменной окружения OPENAI_API_KEY).
        :param openai_url: URL API OpenAI (если не задан, используется значение по умолчанию).
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logging.error(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass it to the constructor.")
            raise ValueError("OpenAI API key is missing.")

        self.openai_url = openai_url or os.getenv(
            "OPENAI_URL") or "https://api.openai.com/v1/chat/completions"  # Added default URL
        if not self.openai_url:
            logging.warning("OpenAI URL not found. Using default: https://api.openai.com/v1/chat/completions")

        self.headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }

        self.model = "gpt-4o"
        logging.info("OpenAI class initialized.")

    def get_response(self, user_prompt: str, system_prompt: str = "") -> Optional[str]:
        """
        Получает ответ от OpenAI API.
        :param user_prompt: Запрос пользователя.
        :param system_prompt: Системное сообщение для контекста (опционально).
        :return: Ответ от OpenAI или None в случае ошибки.
        """
        payload = {
            "model": self.model,  # Now use "gpt-4o" or other model you like
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

        try:
            logging.info(f"Sending request to OpenAI API: {self.openai_url}")
            response = requests.post(self.openai_url, headers=self.headers, json=payload, timeout=10)  # Adding timeout
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            json_response = response.json()

            if json_response.get("choices") and len(
                    json_response["choices"]) > 0:  #Use get method in case element is missing
                content = json_response["choices"][0]["message"]["content"].strip()
                logging.info("Successfully got response from OpenAI")
                return content
            else:
                logging.warning("No choices in OpenAI response.")
                return None

        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {e}")
            return None
