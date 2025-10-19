# wrapper/providers/mistral_provider.py
from typing import List, Dict, Any, Union, Iterator
from mistralai import Mistral, MistralError
from wrapper.base import BaseLLM
from wrapper.utils import get_or_request_key, ColorLogger
from wrapper.config import *

log = ColorLogger(enable_debug=SHOW_LOGS)

class MistralProvider(BaseLLM):
    """
    Official Mistral SDK (v1+).
    Chat: client.chat.complete()
    Stream: client.chat.stream()
    """

    def __init__(self):
        self.api_key = get_or_request_key("MISTRAL_API_KEY", "Please enter your Mistral API Key")
        self.client = Mistral(api_key=self.api_key)
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
                resp = self.client.chat.complete(
                    model=model,
                    messages=final_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    **kwargs,
                )
                return (resp.choices[0].message.content or "").strip()
            def _stream() -> Iterator[str]:
                s = self.client.chat.stream(
                    model=model,
                    messages=final_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    **kwargs,
                )
                for chunk in s:
                    delta = getattr(chunk.choices[0].delta, "content", None)
                    if delta:
                        yield delta
            return _stream()

        except MistralError as e:
            log.error(f"[mistral] API error: {e}")
            raise RuntimeError(str(e))
        except Exception as e:
            log.error(f"[mistral] unexpected: {e}")
            raise RuntimeError(str(e))

    def list_models(self) -> List[str]:
        try:
            r = self.client.models.list()
            return [m.id for m in getattr(r, "data", [])]
        except Exception as e:
            log.error(f"[mistral] list_models failed: {e}")
            return []
