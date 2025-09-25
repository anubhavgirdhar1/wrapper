import sys
import re
import requests
from openai import AzureOpenAI, AuthenticationError, APITimeoutError, APIError
from wrapper.base import BaseLLM
from wrapper.utils import get_or_request_key, ColorLogger
from wrapper.config import *

log = ColorLogger(enable_debug=SHOW_LOGS)

class AzureProvider(BaseLLM):
    def __init__(self):
        # Auto fetch API keys and endpoint from env or ask user
        self.api_key = get_or_request_key("AZURE_OPENAI_API_KEY", "Please enter your Azure OpenAI API Key")
        self.endpoint = get_or_request_key("AZURE_OPENAI_ENDPOINT", "Please enter your Azure OpenAI Endpoint (e.g., https://your-resource.openai.azure.com)")
        self.api_version = get_or_request_key("AZURE_OPENAI_API_VERSION", "Please enter Azure OpenAI API Version (e.g., 2024-10-21)")

        # Clean endpoint URL
        if not self.endpoint.startswith(('http://', 'https://')):
            self.endpoint = f"https://{self.endpoint}"
        self.endpoint = self.endpoint.rstrip('/')

        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            api_version=self.api_version,
            timeout=30.0
        )
        self.default_system_prompt = "You are a helpful AI assistant."

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

        try:
            if not stream:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=final_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    **kwargs
                )
                return response.choices[0].message.content.strip()
            else:
                output = []
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
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        sys.stdout.write(content)
                        sys.stdout.flush()
                        output.append(content)
                return "".join(output).strip()

        except AuthenticationError as e:
            log.error(f"Authentication failed: {str(e)}")
            raise RuntimeError("Invalid API key or endpoint. Please check .env file.")
        except APITimeoutError as e:
            log.error(f"API request timed out: {str(e)}")
            raise RuntimeError("Request timed out. Please try again.")
        except APIError as e:
            log.error(f"Azure OpenAI API error: {str(e)}")
            raise RuntimeError(f"API error: {str(e)}")
        except Exception as e:
            log.error(f"Unexpected error: {str(e)}")
            raise RuntimeError(f"Unexpected error occurred: {str(e)}")

    def list_models(self, **kwargs) -> list[str]:
        """Dynamically fetch available models from Azure OpenAI API"""
        url = f"{self.endpoint}/openai/models"
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        params = {
            "api-version": self.api_version
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            models_data = response.json().get("data", [])

            if models_data:
                model_ids = []

                if SHOW_LOGS:
                    log.info("\nAvailable Azure OpenAI models:")

                for idx, model in enumerate(models_data, start=1):
                    model_id = model.get("id", "unknown")
                    model_ids.append(model_id)

                    if SHOW_LOGS:
                        status = model.get("status", "unknown")
                        capabilities = model.get("capabilities", {})
                        chat_completion = capabilities.get("chat_completion", False)
                        completion = capabilities.get("completion", False)
                        embeddings = capabilities.get("embeddings", False)

                        cap_str = []
                        if chat_completion:
                            cap_str.append("Chat")
                        if completion:
                            cap_str.append("Completion")
                        if embeddings:
                            cap_str.append("Embeddings")

                        log.info(f"  {idx:02d}. {model_id} | Status: {status} | Capabilities: {', '.join(cap_str) or 'None'}")

                return model_ids
            else:
                log.warning("No models found from API!")

        except requests.RequestException as e:
            log.error(f"Failed to retrieve models from API: {e}")
