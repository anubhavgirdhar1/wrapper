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
        log.debug("[groq] Initializing GroqProvider")
        self.api_key = get_or_request_key("GROQ_API_KEY", "Please enter your Groq API Key")
        self.client = Groq(api_key=self.api_key)
        self.default_system_prompt = "You are a helpful AI assistant."
        log.success("[groq] GroqProvider initialized successfully")

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
        log.info(f"[groq] Starting generation with model: {model}")
        log.debug(f"[groq] Stream mode: {stream}, Temperature: {temperature}, Max tokens: {max_tokens}")

        if messages and isinstance(messages, list):
            final_messages = messages
            log.debug(f"[groq] Using provided messages (count: {len(messages)})")
        else:
            final_messages = [
                {"role": "system", "content": self.default_system_prompt},
                {"role": "user", "content": prompt or "Hello"},
            ]
            log.debug(f"[groq] Generated default messages with prompt: {prompt[:50] if prompt else 'Hello'}...")

        api_params = {
            "model": model,
            "messages": final_messages,
            "stream": stream,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        api_params.update({k: v for k, v in kwargs.items() if v is not None})
        log.debug(f"[groq] API parameters prepared: {list(api_params.keys())}")

        try:
            log.debug("[groq] Sending request to Groq API")
            response = self.client.chat.completions.create(**api_params)

            if stream:
                log.info("[groq] Streaming response initiated")
                def stream_generator() -> Iterator[str]:
                    chunk_count = 0
                    for chunk in response:
                        content = chunk.choices[0].delta.content
                        if content:
                            chunk_count += 1
                            yield content
                    log.debug(f"[groq] Streaming completed with {chunk_count} chunks")
                return stream_generator()

            result = (response.choices[0].message.content or "").strip()
            log.success(f"[groq] Generation completed successfully (length: {len(result)} chars)")
            return result

        except AuthenticationError as e:
            log.error(f"[groq] Authentication failed - verify your API key in .env")
            log.debug(f"[groq] AuthenticationError details: {e}")
            raise RuntimeError(f"Authentication error: {str(e)}")
        except RateLimitError as e:
            log.error(f"[groq] Rate limit exceeded - please wait before retrying")
            log.debug(f"[groq] RateLimitError details: {e}")
            raise RuntimeError(f"Rate limit error: {str(e)}")
        except NotFoundError as e:
            log.error(f"[groq] Model '{model}' not found - check available models")
            log.debug(f"[groq] NotFoundError details: {e}")
            raise RuntimeError(f"Model not found: {str(e)}")
        except BadRequestError as e:
            log.error(f"[groq] Invalid request parameters")
            log.debug(f"[groq] BadRequestError details: {e}")
            raise RuntimeError(f"Bad request: {str(e)}")
        except (
            PermissionDeniedError,
            UnprocessableEntityError,
            InternalServerError,
            APIConnectionError
        ) as e:
            log.error(f"[groq] API error: {e.__class__.__name__}")
            log.debug(f"[groq] Error details: {e}")
            raise RuntimeError(str(e))
        except Exception as e:
            log.error(f"[groq] Unexpected error occurred")
            log.debug(f"[groq] Exception details: {type(e).__name__} - {e}")
            raise RuntimeError(str(e))

    def list_models(self) -> List[str]:
        log.info("[groq] Fetching available models from Groq API")
        try:
            models = self.client.models.list()
            model_ids = [m.id for m in models.data]
            log.success(f"[groq] Successfully retrieved {len(model_ids)} models")
            log.debug(f"[groq] Available models: {', '.join(model_ids)}")
            return model_ids
        except AuthenticationError as e:
            log.error(f"[groq] Authentication failed while listing models")
            log.debug(f"[groq] AuthenticationError: {e}")
            return []
        except RateLimitError as e:
            log.error(f"[groq] Rate limit exceeded while listing models")
            log.debug(f"[groq] RateLimitError: {e}")
            return []
        except APIConnectionError as e:
            log.error(f"[groq] Connection error while listing models")
            log.debug(f"[groq] APIConnectionError: {e}")
            return []
        except Exception as e:
            log.error(f"[groq] Unexpected error while listing models")
            log.debug(f"[groq] Exception: {type(e).__name__} - {e}")
            return []