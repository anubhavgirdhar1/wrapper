import sys
import re
import requests
from anthropic import Anthropic, AuthenticationError, APITimeoutError, APIError
from ..base import BaseLLM
from ..utils import get_or_request_key, ColorLogger
from ..config import *

log = ColorLogger(enable_debug=SHOW_LOGS)

class AnthropicProvider(BaseLLM):
    def __init__(self):
        # Auto fetch API key from env or ask user
        self.api_key = get_or_request_key("ANTHROPIC_API_KEY", "Please enter your Anthropic API Key")
        self.client = Anthropic(api_key=self.api_key, timeout=30.0)  # Added timeout
        self.default_system_prompt = "You are a helpful AI assistant."
        self.base_url = "https://api.anthropic.com/v1"

    def _sanitize_input(self, text: str) -> str:
        """Sanitize user input to prevent potential issues"""
        if not isinstance(text, str):
            return str(text)
        # Remove potentially harmful characters while preserving functionality
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        return sanitized.strip()

    def _validate_messages(self, messages: list) -> bool:
        """Validate message structure"""
        if not isinstance(messages, list):
            return False
        for msg in messages:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                return False
            if msg['role'] not in ['system', 'user', 'assistant']:
                return False
        return True

    def generate(
        self, 
        model: str,
        prompt: str = None,
        messages: list[dict] = None,
        temperature: float = 0.7, 
        max_tokens: int = 1024, 
        top_p: float = 0.1, 
        stream: bool = False, 
        **kwargs
    ) -> str:
        if not model or not isinstance(model, str):
            raise ValueError("Model name must be a non-empty string")

        if prompt:
            prompt = self._sanitize_input(prompt)

        if messages:
            if not self._validate_messages(messages):
                raise ValueError("Invalid message format")
            final_messages = []
            for msg in messages:
                sanitized_msg = {
                    'role': msg['role'],
                    'content': self._sanitize_input(msg['content'])
                }
                final_messages.append(sanitized_msg)
        else:
            system_default = {"role": "system", "content": self.default_system_prompt}
            user_msg = {"role": "user", "content": prompt or "Hello"}
            final_messages = [system_default, user_msg]

        system_content = ""
        user_messages = []

        for msg in final_messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                user_messages.append(msg)
        try:
            if not stream:
                response = self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    system=system_content or self.default_system_prompt,
                    messages=user_messages,
                    **kwargs
                )
                return response.content[0].text.strip()
            else:
                output = []
                with self.client.messages.stream(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    system=system_content or self.default_system_prompt,
                    messages=user_messages,
                    **kwargs
                ) as stream_resp:
                    for event in stream_resp:
                        if hasattr(event, "type") and event.type == "content_block_delta":
                            delta_content = getattr(event, "delta", {}).get("text", "")
                            if delta_content:
                                sys.stdout.write(delta_content)
                                sys.stdout.flush()
                                output.append(delta_content)
                return "".join(output).strip()
        except AuthenticationError as e:
            log.error(f"Authentication failed: {str(e)}")
            raise RuntimeError("Invalid API key. Please check .env file.")
        except APITimeoutError as e:
            log.error(f"API request timed out: {str(e)}")
            raise RuntimeError("Request timed out. Please try again.")
        except APIError as e:
            log.error(f"Anthropic API error: {str(e)}")
            raise RuntimeError(f"API error: {str(e)}")
        except Exception as e:
            log.error(f"Unexpected error: {str(e)}")
            raise RuntimeError(f"Unexpected error occurred: {str(e)}")

    def list_models(self, **kwargs) -> list[str]:
        """Dynamically fetch available models from Anthropic API"""
        url = f"{self.base_url}/models"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }

        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            models_data = response.json().get("data", [])

            if models_data:
                model_ids = [model.get("id", "") for model in models_data if model.get("id")]

                if SHOW_LOGS:
                    log.info("\nAvailable Anthropic models:")
                    for idx, model in enumerate(models_data, start=1):
                        model_id = model.get("id", "unknown")
                        display_name = model.get("display_name", model_id)
                        created_at = model.get("created_at", "unknown")
                        log.info(f"  {idx:02d}. {display_name} | ID: {model_id} | Created: {created_at}")

                return model_ids
            else:
                log.warning("No models found from API!")

        except requests.RequestException as e:
            log.error(f"Failed to retrieve models from API: {e}")
