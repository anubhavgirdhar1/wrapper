from typing import List, Dict, Any, Union, Iterator
from openai import OpenAI, AuthenticationError, APIError, APITimeoutError, APIConnectionError, RateLimitError
from wrapper.base import BaseLLM
from wrapper.utils import get_or_request_key, ColorLogger
from wrapper.config import *

log = ColorLogger(enable_debug=SHOW_LOGS)

class HuggingFaceProvider(BaseLLM):
    def __init__(self):
        self.api_key = get_or_request_key("HUGGINGFACE_API_KEY", "Please enter your Hugging Face API Key")
        self.client = OpenAI(api_key=self.api_key, base_url="https://router.huggingface.co/v1")
        self.default_system_prompt = "You are a helpful AI assistant."

    def generate(
        self,
        model: str,
        prompt: str = None,
        messages: List[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_tokens: int = 200,
        top_p: float = 0.95,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Iterator[str]]:
        if messages is None:
            final_messages = [
                {"role": "system", "content": self.default_system_prompt},
                {"role": "user", "content": prompt or "Hello"},
            ]
        else:
            final_messages = messages

        try:
            if not stream:
                r = self.client.chat.completions.create(
                    model=model,
                    messages=final_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    **kwargs
                )
                return (r.choices[0].message.content or "").strip()

            def stream_gen():
                stream_response = self.client.chat.completions.create(
                    model=model,
                    messages=final_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stream=True,
                    **kwargs
                )
                for chunk in stream_response:
                    delta = getattr(chunk.choices[0].delta, "content", None)
                    if delta:
                        yield delta

            return stream_gen()

        except AuthenticationError as e:
            log.error(f"Hugging Face auth failed: {e}")
            raise RuntimeError("Invalid Hugging Face API key.")
        except APITimeoutError as e:
            log.error(f"Hugging Face timeout: {e}")
            raise RuntimeError("Hugging Face request timed out.")
        except (APIError, RateLimitError, APIConnectionError) as e:
            log.error(f"Hugging Face API error: {e}")
            raise RuntimeError(f"Hugging Face API error: {e}")
        except Exception as e:
            log.error(f"Hugging Face unexpected error: {e}")
            raise RuntimeError(f"Unexpected error: {e}")

    def list_models(self) -> List[str]:
        try:
            models = self.client.models.list().data
            chat_models = [m.id for m in models if 'instruct' in m.id.lower() or 'chat' in m.id.lower()]
            return chat_models[:50]
        except Exception as e:
            log.error(f"Hugging Face list_models failed: {e}")
            return []