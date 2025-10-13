from typing import List, Dict, Any, Iterable
from wrapper.base import BaseLLM
from wrapper.utils import get_or_request_key, ColorLogger
from wrapper.config import *
from google import genai
from google.genai import types

log = ColorLogger(enable_debug=SHOW_LOGS)

class GeminiProvider(BaseLLM):
    def __init__(self):
        self.api_key = get_or_request_key("GEMINI_API_KEY", "Please enter your Gemini API Key")
        self.client = genai.Client(api_key=self.api_key)
        self.default_system_prompt = "You are a helpful AI assistant."

    def _messages_to_contents(self, messages: List[Dict[str, Any]]) -> List[types.Content]:
        if not messages:
            return [types.Content(role="user", parts=[types.Part.from_text(text="Hello")])]
        system_buf: List[str] = []
        out: List[types.Content] = []
        for m in messages:
            role = (m.get("role") or "").lower()
            text = str(m.get("content") or "")
            if role == "system":
                system_buf.append(text)
                continue
            mapped = "model" if role == "assistant" else "user"
            if system_buf and mapped == "user":
                text = f"<SYSTEM>\n{'\n'.join(system_buf)}\n</SYSTEM>\n{text}"
                system_buf.clear()
            out.append(types.Content(role=mapped, parts=[types.Part.from_text(text=text)]))
        if system_buf and not out:
            out.append(types.Content(role="user",parts=[types.Part.from_text(text=f"<SYSTEM>\n{'\n'.join(system_buf)}\n</SYSTEM>")]))
        return out

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
        contents = self._messages_to_contents(messages) if messages else self._messages_to_contents([
            {"role": "system", "content": self.default_system_prompt},
            {"role": "user", "content": prompt or "Hello"},
        ])
        cfg = types.GenerateContentConfig(temperature=temperature, top_p=top_p, max_output_tokens=max_tokens)

        if not stream:
            r = self.client.models.generate_content(model=model, contents=contents, config=cfg)
            return (r.text or "").strip()

        chunks: List[str] = []
        for part in self.client.models.generate_content_stream(model=model, contents=contents, config=cfg):
            txt = getattr(part, "text", None)
            if txt:
                chunks.append(txt)
        return "".join(chunks).strip()

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
        contents = self._messages_to_contents(messages) if messages else self._messages_to_contents([
            {"role": "system", "content": self.default_system_prompt},
            {"role": "user", "content": prompt or "Hello"},
        ])
        cfg = types.GenerateContentConfig(temperature=temperature, top_p=top_p, max_output_tokens=max_tokens)
        for part in self.client.models.generate_content_stream(model=model, contents=contents, config=cfg):
            txt = getattr(part, "text", None)
            if txt:
                yield txt

    def list_models(self) -> List[str]:
        try:
            names: List[str] = []
            for m in self.client.models.list():
                mid = getattr(m, "name", None) or getattr(m, "model", None)
                if mid:
                    names.append(mid)
            return names
        except Exception as e:
            log.error(f"Gemini list_models failed: {e}")
            return []
