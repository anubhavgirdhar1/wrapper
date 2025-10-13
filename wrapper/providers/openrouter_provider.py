from typing import List, Dict, Any, Iterable
import requests
from openai import OpenAI, AuthenticationError, APIError, APITimeoutError
from wrapper.base import BaseLLM
from wrapper.utils import get_or_request_key, ColorLogger
from wrapper.config import *

log = ColorLogger(enable_debug=SHOW_LOGS)

class OpenRouterProvider(BaseLLM):
    def __init__(self):
        self.api_key = get_or_request_key("OPENROUTER_API_KEY", "Please enter your OpenRouter API Key")
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.api_key)
        self.default_system_prompt = "You are a helpful AI assistant."
        self.base_api_url = "https://openrouter.ai/api/v1"

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
    ) -> str:
        final_messages = messages or [
            {"role": "system", "content": self.default_system_prompt},
            {"role": "user", "content": prompt or "Hello"},
        ]
        try:
            if not stream:
                r = self.client.chat.completions.create(
                    model=model, messages=final_messages,
                    temperature=temperature, max_tokens=max_tokens, top_p=top_p, **kwargs
                )
                return (r.choices[0].message.content or "").strip()

            out: List[str] = []
            for chunk in self.client.chat.completions.create(
                model=model, messages=final_messages,
                temperature=temperature, max_tokens=max_tokens, top_p=top_p,
                stream=True, **kwargs
            ):
                delta = getattr(chunk.choices[0].delta, "content", None)
                if delta:
                    out.append(delta)
            return "".join(out).strip()

        except AuthenticationError as e:
            log.error(f"OpenRouter auth failed: {e}")
            raise RuntimeError("Invalid OpenRouter API key.")
        except APITimeoutError as e:
            log.error(f"OpenRouter timeout: {e}")
            raise RuntimeError("OpenRouter request timed out.")
        except APIError as e:
            log.error(f"OpenRouter API error: {e}")
            raise RuntimeError(f"OpenRouter API error: {e}")
        except Exception as e:
            log.error(f"OpenRouter unexpected error: {e}")
            raise RuntimeError(f"Unexpected error: {e}")

    def generate_stream(
        self,
        model: str,
        prompt: str = None,
        messages: List[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_tokens: int = 200,
        top_p: float = 0.1,
        **kwargs
    ) -> Iterable[str]:
        final_messages = messages or [
            {"role": "system", "content": self.default_system_prompt},
            {"role": "user", "content": prompt or "Hello"},
        ]
        for chunk in self.client.chat.completions.create(
            model=model, messages=final_messages,
            temperature=temperature, max_tokens=max_tokens, top_p=top_p,
            stream=True, **kwargs
        ):
            delta = getattr(chunk.choices[0].delta, "content", None)
            if delta:
                yield delta

    def list_models(self) -> List[str]:
        try:
            r = requests.get(f"{self.base_api_url}/models",
                             headers={"Authorization": f"Bearer {self.api_key}",
                                      "Content-Type": "application/json"},
                             timeout=30)
            r.raise_for_status()
            data = r.json() or {}
            return [m.get("id") for m in (data.get("data") or []) if m.get("id")]
        except Exception as e:
            log.error(f"OpenRouter list_models failed: {e}")
            return []
