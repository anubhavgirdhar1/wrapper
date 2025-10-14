# wrapper/providers/groq_provider.py
from typing import List, Dict, Any, Iterator, Union
from wrapper.base import BaseLLM
from wrapper.utils import get_or_request_key, ColorLogger
from wrapper.config import *
from groq import (
    Groq,
    BadRequestError,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    UnprocessableEntityError,
    RateLimitError,
    InternalServerError,
    APIConnectionError,
)

log = ColorLogger(enable_debug=SHOW_LOGS)

class GroqProvider(BaseLLM):
    def __init__(self):
        self.api_key = get_or_request_key("GROQ_API_KEY", "Please enter your Groq API Key")
        self.client = Groq(api_key=self.api_key)
        self.default_system_prompt = "You are a helpful AI assistant."

    def generate(
        self,
        model: str,
        prompt: str = None,
        messages: List[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_tokens: int = 200,
        top_p: float = 0.1,
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
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        api_params.update({k: v for k, v in kwargs.items() if v is not None})

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

        except (
            AuthenticationError,
            PermissionDeniedError,
            NotFoundError,
            UnprocessableEntityError,
            RateLimitError,
            InternalServerError,
            APIConnectionError,
            BadRequestError
        ) as e:
            log.error(f"[groq] API error: {e.__class__.__name__} - {e}")
            raise RuntimeError(str(e))
        except Exception as e:
            log.error(f"[groq] Unexpected error: {e}")
            raise RuntimeError(str(e))

    def list_models(self) -> List[str]:
        try:
            models = self.client.models.list()
            return [m.id for m in models.data]
        except (APIConnectionError, RateLimitError, AuthenticationError) as e:
            log.error(f"[groq] list_models API error: {e}")
            return []
        except Exception as e:
            log.error(f"[groq] list_models unexpected error: {e}")
            return []