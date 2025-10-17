# wrapper/providers/xai_provider.py
from typing import List, Dict, Any, Union, Iterator
from openai import (
    OpenAI,
    AuthenticationError,
    APIError,
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
)
from wrapper.base import BaseLLM
from wrapper.utils import get_or_request_key, ColorLogger
from wrapper.config import *

log = ColorLogger(enable_debug=SHOW_LOGS)

class XAIProvider(BaseLLM):
    def __init__(self):
        self.api_key = get_or_request_key("XAI_API_KEY", "Please enter your xAI API Key")
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.x.ai/v1")
        self.default_system_prompt = "You are a helpful AI assistant."
    def generate(
        self,
        model: str,
        prompt: str = None,
        messages: List[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_tokens: int = 200,
        top_p: float = 0.70, 
        stream: bool = False,
        **kwargs
    ) -> Union[str, Iterator[str]]:
        final_messages = (
            messages
            if messages and isinstance(messages, list)
            else [
                {"role": "system", "content": self.default_system_prompt},
                {"role": "user", "content": prompt or "Hello"},
            ])
        try:
            if not stream:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=final_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    **kwargs,
                )
                return (resp.choices[0].message.content or "").strip()
            def _stream() -> Iterator[str]:
                s = self.client.chat.completions.create(
                    model=model,
                    messages=final_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stream=True,
                    **kwargs,
                )
                for chunk in s:
                    delta = getattr(chunk.choices[0].delta, "content", None)
                    if delta:
                        yield delta
            return _stream()

        except AuthenticationError as e:
            log.error(f"[xai] auth failed: {e}")
            raise RuntimeError("Invalid xAI API key.")
        except APITimeoutError as e:
            log.error(f"[xai] timeout: {e}")
            raise RuntimeError("xAI request timed out.")
        except (APIConnectionError, RateLimitError, APIError) as e:
            log.error(f"[xai] API error: {e}")
            raise RuntimeError(str(e))
        except Exception as e:
            log.error(f"[xai] unexpected: {e}")
            raise RuntimeError(str(e))

    def list_models(self) -> List[str]:
        try:
            r = self.client.models.list()
            return [m.id for m in r.data]
        except Exception as e:
            log.error(f"[xai] list_models failed: {e}")
            return []
