# wrapper/providers/gemini_provider.py
from typing import List, Dict, Any, Iterator, Union
from google import genai
from google.genai import types as gtypes
from google.genai import errors as google_exceptions
from wrapper.base import BaseLLM
from wrapper.utils import get_or_request_key, ColorLogger
from wrapper.config import *

log = ColorLogger(enable_debug=SHOW_LOGS)

class GeminiProvider(BaseLLM):
    def __init__(self):
        self.api_key = get_or_request_key("GEMINI_API_KEY", "Please enter your Gemini API Key")
        self.client = genai.Client(api_key=self.api_key)
        self.default_system_prompt = "You are a helpful AI assistant."

    def generate(
        self,
        model: str,
        prompt: str = None,
        messages: List[Dict[str, Any]] = None,
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        **kwargs
    ) -> Union[str, Iterator[str]]:
        
        source_messages = messages if messages and isinstance(messages, list) else [
            {"role": "system", "content": self.default_system_prompt},
            {"role": "user", "content": prompt or "Hello"},
        ]

        sys_parts = [msg.get("content", "") for msg in source_messages if msg.get("role") == "system"]
        system_text = "\n".join(p.strip() for p in sys_parts if p).strip() or None
        
        contents = []
        for msg in source_messages:
            role = (msg.get("role") or "").lower()
            text = str(msg.get("content") or "")
            if role == "system":
                continue
            elif role in ("assistant", "model"):
                contents.append(gtypes.Content(role="model", parts=[gtypes.Part.from_text(text=text)]))
            else:
                contents.append(gtypes.Content(role="user", parts=[gtypes.Part.from_text(text=text)]))

        config = gtypes.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
            system_instruction=system_text,
            **{k: v for k, v in kwargs.items() if v is not None}
        )
        
        try:
            if stream:
                response_stream = self.client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=config
                )
                def stream_generator() -> Iterator[str]:
                    for chunk in response_stream:
                        if text := getattr(chunk, "text", None):
                            yield text
                return stream_generator()
            
            response = self.client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
            return (getattr(response, "text", "") or "").strip()

        except google_exceptions.InvalidArgument as e:
            log.error(f"[gemini] Invalid Argument (400): The request is malformed. Check parameters. Details: {e}")
            raise RuntimeError(str(e))
        except google_exceptions.PermissionDenied as e:
            log.error(f"[gemini] Permission Denied (403): Check your API key and permissions. Details: {e}")
            raise RuntimeError(str(e))
        except google_exceptions.NotFound as e:
            log.error(f"[gemini] Not Found (404): The requested resource (e.g., model) was not found. Details: {e}")
            raise RuntimeError(str(e))
        except google_exceptions.ResourceExhausted as e:
            log.error(f"[gemini] Resource Exhausted (429): You have exceeded your rate limit. Details: {e}")
            raise RuntimeError(str(e))
        except (google_exceptions.InternalServerError, google_exceptions.ServiceUnavailable) as e:
            log.error(f"[gemini] Server Error (500/503): The service is unavailable or encountered an internal error. Please retry. Details: {e}")
            raise RuntimeError(str(e))
        except Exception as e:
            log.error(f"[gemini] An unexpected error occurred: {e}")
            raise RuntimeError(str(e))

    def list_models(self) -> List[str]:
        try:
            return [getattr(m, "name", None) or getattr(m, "model", None) for m in self.client.models.list()]
        except (google_exceptions.PermissionDenied, google_exceptions.ResourceExhausted) as e:
            log.error(f"[gemini] list_models API error: {e}")
            return []
        except Exception as e:
            log.error(f"[gemini] list_models unexpected error: {e}")
            return []