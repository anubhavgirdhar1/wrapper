from wrapper.utils import get_or_request_key, ColorLogger
from wrapper.base import BaseProvider
import requests
from typing import Generator, Optional, Dict, Any

class MistralProvider(BaseProvider):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.logger = ColorLogger("MistralProvider")
        self.api_key = get_or_request_key("MISTRAL_API_KEY", config)
        self.base_url = config.get("base_url", "https://api.mistral.ai/v1") if config else "https://api.mistral.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def list_models(self) -> Dict[str, Any]:
        url = f"{self.base_url}/models"
        self.logger.info(f"Listing models from {url}")
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def generate(self, prompt: str, model: str, stream: bool = False, **kwargs) -> Any:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs
        }
        if stream:
            payload["stream"] = True
            self.logger.info(f"Streaming completion from {url} for model {model}")
            return self._streaming_request(url, payload)
        else:
            self.logger.info(f"Non-streaming completion from {url} for model {model}")
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()

    def _streaming_request(self, url: str, payload: dict) -> Generator[str, None, None]:
        with requests.post(url, headers=self.headers, json=payload, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    yield line.decode("utf-8")
