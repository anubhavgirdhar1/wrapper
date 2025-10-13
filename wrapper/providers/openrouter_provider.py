import sys
import requests
from openai import OpenAI, AuthenticationError, APIError, APITimeoutError
from wrapper.base import BaseLLM
from wrapper.utils import get_or_request_key, get_key_silent, ColorLogger
from wrapper.config import *

log = ColorLogger(enable_debug=SHOW_LOGS)

class OpenRouterProvider(BaseLLM):
    """
    OpenRouter provider using the OpenAI SDK pointed at OpenRouter's base URL.
    Keeps the same generate() contract and streaming behavior as other providers.
    """
    def __init__(self):
        # Fetch API key (prompt if missing, same pattern as other providers)
        self.api_key = get_or_request_key("OPENROUTER_API_KEY", "Please enter your OpenRouter API Key")

        # Optional attribution headers (do NOT prompt; purely optional)
        self.site_url = (get_key_silent("OPENROUTER_SITE_URL") or "").strip()
        self.app_title = (get_key_silent("OPENROUTER_APP_TITLE") or "").strip()

        # OpenAI SDK, pointed at OpenRouter
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

        self.default_system_prompt = "You are a helpful AI assistant."
        self.base_api_url = "https://openrouter.ai/api/v1"

        # Precompute optional extra headers for leaderboard attribution
        self.extra_headers = {}
        if self.site_url:
            self.extra_headers["HTTP-Referer"] = self.site_url
        if self.app_title:
            self.extra_headers["X-Title"] = self.app_title

    def generate(
        self, 
        model: str,
        prompt: str = None,
        messages: list[dict] = None,
        temperature: float = 0.7, 
        max_tokens: int = 200, 
        top_p: float = 0.1, 
        stream: bool = False, 
        **kwargs
    ) -> str:
        """
        Match the repo's convention:
        - If messages provided, pass-through.
        - Else, build [system, user] messages like other providers.
        - Streaming prints to stdout and returns the full string.
        """
        if messages:
            final_messages = messages
        else:
            system_default = {"role": "system", "content": self.default_system_prompt}
            user_msg = {"role": "user", "content": prompt or "Hello"}
            final_messages = [system_default, user_msg]

        try:
            if not stream:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=final_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    # OpenRouter attribution headers (optional)
                    extra_headers=self.extra_headers if self.extra_headers else None,
                    **kwargs
                )
                return (resp.choices[0].message.content or "").strip()
            else:
                output = []
                stream_resp = self.client.chat.completions.create(
                    model=model,
                    messages=final_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stream=True,
                    extra_headers=self.extra_headers if self.extra_headers else None,
                    **kwargs
                )
                for chunk in stream_resp:
                    # Same pattern as groq_provider streaming
                    delta = getattr(chunk.choices[0].delta, "content", None)
                    if delta:
                        sys.stdout.write(delta)
                        sys.stdout.flush()
                        output.append(delta)
                return "".join(output).strip()

        except AuthenticationError as e:
            log.error(f"Authentication failed: {str(e)}")
            raise RuntimeError("Invalid API key for OpenRouter. Please check your .env file.")
        except APITimeoutError as e:
            log.error(f"API request timed out: {str(e)}")
            raise RuntimeError("Request timed out. Please try again.")
        except APIError as e:
            log.error(f"OpenRouter API error: {str(e)}")
            raise RuntimeError(f"API error: {str(e)}")
        except Exception as e:
            log.error(f"Unexpected error: {str(e)}")
            raise RuntimeError(f"Unexpected error occurred: {str(e)}")

    def list_models(self) -> list[str]:
        """
        Fetch available models from OpenRouter:
        GET /api/v1/models  (requires Authorization)
        """
        url = f"{self.base_api_url}/models"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json() or {}
            models_data = data.get("data", []) or []

            model_ids = [m.get("id") for m in models_data if m.get("id")]

            if SHOW_LOGS and models_data:
                log.info("\nAvailable OpenRouter Models:")
                for i, m in enumerate(models_data, 1):
                    mid = m.get("id", "unknown")
                    name = m.get("name", mid)
                    ctx = m.get("context_length", "N/A")
                    log.info(f"  {i:02d}. {name} | ID: {mid} | Context: {ctx}")

            return model_ids
        except requests.RequestException as e:
            log.error(f"Failed to retrieve models from OpenRouter API: {e}")
            return []
        except Exception as e:
            log.error(f"Error parsing OpenRouter models: {e}")
            return []
