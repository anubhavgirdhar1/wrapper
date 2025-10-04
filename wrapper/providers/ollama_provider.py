import sys
import requests
from wrapper.base import BaseLLM
from wrapper.utils import set_key, get_key_silent, ColorLogger
from pathlib import Path
from dotenv import load_dotenv
import json
from wrapper.config import *

log = ColorLogger(enable_debug=SHOW_LOGS)

class OllamaProvider(BaseLLM):
    DEFAULT_PORT = 11434  # fixed port

    def __init__(self):
        # Use consistent location in user's home directory
        env_file = Path.home() / ".wrapper" / ".env"
        env_file.parent.mkdir(exist_ok=True)
        
        if env_file.exists():
            load_dotenv(env_file)
            log.debug(f"yooo found ur .env chillin at {env_file}")
        else:
            env_file.touch()
            load_dotenv(env_file)
            log.debug(f"no .env? bruh... made u a fresh one at {env_file}")

        # Load values
        self.api_key = (get_key_silent("OLLAMA_API_KEY") or "").strip()
        self.base_url = (get_key_silent("OLLAMA_BASE_URL") or "").strip()
        log.debug(f"api key status: {'secured ✓' if self.api_key else 'missing in action lmao'}")
        log.debug(f"base url vibes: {self.base_url or 'nowhere to be found chief'}")
        
        # Only show menu if base_url is not already configured
        if not self.base_url:
            print("\nOllama Configuration (first time setup):")
            print("1: Provide API Key (optional, for remote Ollama)")
            print("2: Provide Endpoint Host (default: localhost:11434)")
            print("Press Enter to use default localhost")
            choice = input("Select option [1/2 or Enter]: ").strip()

            if choice == "1":
                new_key = input("Enter Ollama API Key: ").strip()
                if new_key:
                    set_key(str(env_file), "OLLAMA_API_KEY", new_key)
                    self.api_key = new_key
                    log.debug("api key locked n loaded")
            elif choice == "2":
                host = input("Enter Ollama API Endpoint Host (e.g., localhost or 192.168.1.100): ").strip()
                if host:
                    self.base_url = f"http://{host}:{self.DEFAULT_PORT}"
                    set_key(str(env_file), "OLLAMA_BASE_URL", self.base_url)
                    log.debug(f"endpoint saved: {self.base_url}")
            
            # Set and save default if nothing was provided
            if not self.base_url:
                self.base_url = f"http://localhost:{self.DEFAULT_PORT}"
                set_key(str(env_file), "OLLAMA_BASE_URL", self.base_url)
                log.debug("saved default localhost to .env so we dont bug u again")

        # Final fallback
        if not self.base_url:
            self.base_url = f"http://localhost:{self.DEFAULT_PORT}"
            log.debug(f"aight defaultin to localhost party: {self.base_url}")

        log.debug("no api key but fk it we ball, might crash later idk" if not self.api_key else "we got everything, lets cook")

    def list_models(self, **kwargs):
        url = f"{self.base_url}/api/tags"   
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        log.debug("lemme check what models u got installed...")
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
                log.debug(f"puh, found {len(models_raw)} models total")
            else:
                log.warning("No models found!")
                log.info("Download one using: ollama pull <model_name>")
                log.debug("bruh ur ollama is emptier than my brain rn")

        except requests.RequestException as e:
            log.error(f"Failed to retrieve models: {e}")
            log.debug("model fetching went boom 💥")

    def generate(self, messages: list = None, prompt: str = None, stream: bool = False, **kwargs) -> str:
        model = kwargs.pop("model", None)

        if not model:
            log.error("ayo wheres the model name at?? cant do shit without it")
            return ""

        log.debug(f"cookin with model: {model} 👨‍🍳")
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        # Handle messages vs prompt
        if messages:
            log.debug(f"processin {len(messages)} messages... this better be good")
            system_msg = "\n".join(
                m["content"] for m in messages if m.get("role") == "system"
            )
            user_msg = "\n".join(
                m["content"] for m in messages if m.get("role") == "user"
            )
            
            if system_msg and user_msg:
                final_prompt = f"{system_msg}\n\n{user_msg}"
                log.debug("got both system n user msgs, perfectenschlag")
            elif user_msg:
                final_prompt = user_msg
                log.debug("just user msg, keepin it simple")
            else:
                final_prompt = system_msg or ""
                log.debug("only system msg?? weird but ok")
                
        elif prompt:
            log.debug("raw prompt mode activated ezpz")
            final_prompt = prompt
        else:
            log.error("bruh u gave me literally nothing to work with")
            return ""

        log.debug(f"final prompt length: {len(final_prompt)} chars... sendin it")
        payload = {"model": model, "prompt": final_prompt, "stream": False, **kwargs}

        try:
            log.debug(f"hittin up {self.base_url}/api/generate...")
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            log.debug("puh! got 200 back, ollama didnt ghost us")

            # Ollama returns newline-delimited JSON even with stream=false
            collected = []
            line_count = 0
            for line in response.text.strip().split('\n'):
                if line:
                    line_count += 1
                    try:
                        data = json.loads(line)
                        piece = data.get("response", "")
                        if piece:
                            collected.append(piece)
                            if stream:
                                print(piece, end="", flush=True)
                    except json.JSONDecodeError as e:
                        log.debug(f"line {line_count} was garbagio: {line[:40]}...")
                        continue

            log.debug(f"parsed {line_count} lines from response")
            result = "".join(collected).strip()
            
            if result:
                log.debug(f"dih! got {len(result)} chars back, looks solid")
            else:
                log.warning("ollama returned jack shit 😭")
                log.debug("response was emptier than my will to live")
                
            return result

        except requests.HTTPError as e:
            log.error(f"ollama threw hands: {e.response.text if hasattr(e, 'response') else e}")
            log.debug("http error, probably model doesnt exist or sumthin")
            return ""
        except requests.RequestException as e:
            log.error(f"connection ded. is ollama even alive?? {e}")
            log.debug("cant reach ollama, did u forget to run `ollama serve` lmaoo")
            return ""
        except Exception as e:
            log.error(f"something catastrophic happened: {e}")
            log.debug("idk what broke but it broke hard 🔥")
            return ""