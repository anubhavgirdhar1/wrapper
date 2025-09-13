import sys
import requests
from wrapper.base import BaseLLM
from wrapper.utils import set_key, get_key_silent, ColorLogger
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import json
from wrapper.config import *

log = ColorLogger(enable_debug=SHOW_LOGS)

class OllamaProvider(BaseLLM):
    DEFAULT_PORT = 11434  # fixed port

    def __init__(self):
        dotenv_path = find_dotenv()
        env_file = dotenv_path or Path(".env")
        if dotenv_path:
            load_dotenv(dotenv_path)
            log.debug(f"Loaded .env from {dotenv_path}")
        else:
            Path(".env").touch()
            load_dotenv()
            log.debug("Created new .env file and loaded it")

        # siletn loader
        self.api_key = (get_key_silent("OLLAMA_API_KEY") or "").strip()
        self.base_url = (get_key_silent("OLLAMA_BASE_URL") or "").strip()
        log.debug(f"Initial API key: {'FOUND' if self.api_key else 'MISSING'}")
        log.debug(f"Initial Base URL: {self.base_url or 'MISSING'}")
 
        # menu for selection
        if not self.api_key and not self.base_url:
            print("Ollama Configuration Menu:")
            print("1: Provide API Key")
            print("2: Provide Endpoint Host (port will remain 11434)")
            print("Enter to skip.")
            choice = input("Select option [1/2]: ").strip()

            if choice == "1":
                new_key = input("Enter Ollama API Key: ").strip()
                if new_key:
                    set_key(str(env_file), "OLLAMA_API_KEY", new_key)
                    self.api_key = new_key
                    print("API Key saved successfully")
                else:
                    print("No API key entered")
            elif choice == "2":
                host = input("Enter Ollama API Endpoint Host (e.g., localhost or 127.0.0.1): ").strip()
                if host:
                    self.base_url = f"http://{host}:{self.DEFAULT_PORT}"
                    set_key(str(env_file), "OLLAMA_BASE_URL", self.base_url)
                    print(f"Endpoint URL set to {self.base_url}")
                else:
                    log.warning("No host entered, using default later")

        # defaults to local 
        if not self.base_url:
            self.base_url = f"http://localhost:{self.DEFAULT_PORT}"
            log.info(f"Using default endpoint: {self.base_url}")

        # bs warnign - will remove later #TODO
        if not self.api_key:
            log.warning("API key is missing. Requests may fail if API key is required.")

    def list_models(self, **kwargs):
        url = f"{self.base_url}/api/tags"   
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            models_raw = response.json().get("models", [])

            if models_raw:
                log.info("\nInstalled Ollama models:")
                for idx, m in enumerate(models_raw, start=1):
                    name = m.get("name", "unknown")
                    mid = m.get("model", name)
                    size = m.get("size", "unknown")
                    log.info(f"  {idx:02d}. {name}  |  ID: {mid}  |  Size: {size}")
            else:
                log.warning("No models found!")
                log.info("Download one using: ollama pull <model_name>")

        except requests.RequestException as e:
            log.error(f"Failed to retrieve models: {e}")

    def generate(self, prompt: str = None, stream: bool = False, **kwargs) -> str:
        model = kwargs.pop("model", None)
        messages = kwargs.pop("messages", None)

        if not model:
            log.error("No model specified for Ollama")
            return ""

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        # Handle messages vs prompt
        if messages:
            # If messages list is provided, merge into a single prompt
            system_msg = "\n".join(
                m["content"] for m in messages if m["role"] == "system"
            )
            user_msg = "\n".join(
                m["content"] for m in messages if m["role"] == "user"
            )
            final_prompt = f"{system_msg}\n\n{user_msg}".strip()
        else:
            # Default system + user
            system_default = "You are a helpful AI assistant."
            final_prompt = f"{system_default}\n\nUser: {prompt}"

        payload = {"model": model, "prompt": final_prompt, **kwargs}

        try:
            with requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers=headers,
                stream=stream
            ) as response:
                response.raise_for_status()

                if stream:
                    collected = []
                    for line in response.iter_lines():
                        if line:
                            try:
                                data = json.loads(line.decode("utf-8"))
                                piece = data.get("response", "")
                                if piece:
                                    collected.append(piece)
                                if data.get("done", False):
                                    break
                            except Exception:
                                continue
                    log.success("Streaming complete")
                    return "".join(collected).strip()
                else:
                    data = response.json()
                    log.success("Received full response from Ollama API")
                    return data.get("response", data.get("text", "")).strip()

        except requests.HTTPError as e:
            log.error(f"Ollama API error: {e.response.text}")
            sys.exit(1)
        except requests.RequestException as e:
            log.error(f"Request failed - Please ensure Ollama is running. Details: {e}")
            sys.exit(1)
