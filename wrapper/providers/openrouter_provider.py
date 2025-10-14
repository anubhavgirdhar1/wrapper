# openrouter_provider.py
from typing import List, Dict, Any, Iterator, Union, Optional
import requests
from openai import OpenAI, AuthenticationError, APIError, APITimeoutError, APIConnectionError, RateLimitError
from wrapper.base import BaseLLM
from wrapper.utils import get_or_request_key, ColorLogger
from wrapper.config import *

log = ColorLogger(enable_debug=SHOW_LOGS)

class OpenRouterProvider(BaseLLM):
    def __init__(self):
        self.api_key = get_or_request_key("OPENROUTER_API_KEY", "Please enter your OpenRouter API Key")
        self.client = OpenAI(api_key=self.api_key, base_url="https://openrouter.ai/api/v1")
        self.default_system_prompt = "You are a helpful AI assistant."
        self.base_api_url = "https://openrouter.ai/api/v1"

    def generate(
        self,
        model: str,
        prompt: str = None,
        messages: List[Dict[str, Any]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Iterator[str]]:
        if messages and isinstance(messages, list):
            final_messages = messages
        else:
            final_messages = [
                {"role": "system", "content": self.default_system_prompt},
                {"role": "user", "content": prompt or "Hello"},
            ]
        api_params = {
            "model": model,
            "messages": final_messages,
            "stream": stream,
            **{k: v for k, v in kwargs.items() if v is not None}
        }
        try:
            response = self.client.chat.completions.create(**api_params)
            if stream:
                def stream_generator() -> Iterator[str]:
                    for chunk in response:
                        content = chunk.choices[0].delta.content
                        if content:
                            yield content
                return stream_generator()
            return (response.choices[0].message.content or "").strip()
        except (AuthenticationError, APIConnectionError, APITimeoutError, RateLimitError, APIError) as e:
            log.error(f"[openrouter] API error: {e}")
            raise RuntimeError(str(e))
        except Exception as e:
            log.error(f"[openrouter] Unexpected error: {e}")
            raise RuntimeError(str(e))

    def list_models(self) -> List[str]:
        try:
            r = requests.get(
                f"{self.base_api_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            return [m["id"] for m in data.get("data", []) if m.get("id")]
        except Exception as e:
            log.error(f"[openrouter] list_models error: {e}")
            return []
