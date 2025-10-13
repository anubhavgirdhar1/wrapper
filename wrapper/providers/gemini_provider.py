# wrapper/providers/gemini_provider.py

import sys
from typing import List, Dict, Any

from wrapper.base import BaseLLM
from wrapper.utils import get_or_request_key, ColorLogger
from wrapper.config import *

from google import genai
from google.genai import types

log = ColorLogger(enable_debug=SHOW_LOGS)


def _messages_to_contents(messages: List[Dict[str, Any]]) -> List[types.Content]:
    """
    Convert OpenAI-style messages [{role, content}] to google-genai 'contents'.
    - user -> role='user'
    - assistant -> role='model'
    - system -> we prefix into the first user turn to keep things simple/consistent
    """
    if not messages:
        return [types.Content(role="user", parts=[types.Part.from_text("Hello")])]

    contents: List[types.Content] = []
    system_buf: List[str] = []

    for m in messages:
        role = (m.get("role") or "").lower()
        content = m.get("content") or ""

        if role == "system":
            system_buf.append(str(content))
            continue

        if role == "assistant":
            mapped_role = "model"
        else:
            mapped_role = "user"

        if system_buf and mapped_role == "user":
            content = f"<SYSTEM>\n{'\n'.join(system_buf)}\n</SYSTEM>\n{content}"
            system_buf.clear()

        contents.append(
            types.Content(
                role=mapped_role,
                parts=[types.Part.from_text(text=str(content))]
            )
        )

    if system_buf and not contents:
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=f"<SYSTEM>\n{'\n'.join(system_buf)}\n</SYSTEM>")]
            )
        )

    return contents


class GeminiProvider(BaseLLM):
    """
    Gemini provider using the official Google Gen AI SDK.
    - Non-streaming & streaming text generation
    - Model listing
    Matches your repo’s generate() contract and logging patterns.
    """

    def __init__(self):
        self.api_key = get_or_request_key("GEMINI_API_KEY", "Please enter your Gemini API Key")
        self.client = genai.Client(api_key=self.api_key)
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
    ) -> str:
        """
        Unified generate entrypoint:
        - If messages provided, convert to Gemini 'contents'
        - Else, build [system, user] like other providers
        - Streaming prints tokens to stdout and returns the final text
        """
        if messages:
            contents = _messages_to_contents(messages)
        else:
            contents = _messages_to_contents([
                {"role": "system", "content": self.default_system_prompt},
                {"role": "user", "content": prompt or "Hello"}
            ])
        gen_config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
        )

        try:
            if not stream:
                resp = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=gen_config,
                )
                return (resp.text or "").strip()

            output_chunks: List[str] = []
            for chunk in self.client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=gen_config,
            ):
                text = getattr(chunk, "text", None)
                if text:
                    sys.stdout.write(text)
                    sys.stdout.flush()
                    output_chunks.append(text)

            return "".join(output_chunks).strip()

        except Exception as e:
            log.error(f"Gemini API error: {e}")
            raise RuntimeError(f"Gemini API error: {e}")

    def list_models(self) -> List[str]:
        """
        List available base models visible to the key.
        Uses the SDK’s models.list() pager.
        """
        try:
            names: List[str] = []
            pager = self.client.models.list()
            for m in pager:
                mid = getattr(m, "name", None) or getattr(m, "model", None)
                if mid:
                    names.append(mid)

            if SHOW_LOGS and names:
                log.info("\nAvailable Gemini Models:")
                for idx, mid in enumerate(names, 1):
                    log.info(f"  {idx:03d}. {mid}")

            return names

        except Exception as e:
            log.error(f"Failed to list Gemini models: {e}")
            return []
