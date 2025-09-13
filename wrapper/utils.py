import os
import sys
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

def set_key(env_file: str, key: str, value: str):
    env_path = Path(env_file)
    lines = []
    if env_path.exists():
        with env_path.open("r") as f:
            lines = f.readlines()

    key_found = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            key_found = True
            break

    if not key_found:
        lines.append(f"{key}={value}\n")

    with env_path.open("w") as f:
        f.writelines(lines)


def get_or_request_key(env_var_name: str, prompt_message: str) -> str:
    """Fetch key from env, prompt if missing."""
    dotenv_path = find_dotenv()
    if dotenv_path:
        load_dotenv(dotenv_path)
    else:
        Path(".env").touch()
        load_dotenv()
    key = os.getenv(env_var_name)
    if not key:
        print(f"{env_var_name} not found in .env")
        new_key = input(f"{prompt_message} (will be saved in .env): ").strip()
        if not new_key:
            print("No key provided, exiting...")
            sys.exit(1)
        env_file = dotenv_path or Path(".env")
        set_key(str(env_file), env_var_name, new_key)
        os.environ[env_var_name] = new_key
        key = new_key
    return key


def get_key_silent(env_var_name: str):
    """Return env key if exists, else None. No prompts."""
    dotenv_path = find_dotenv()
    if dotenv_path:
        load_dotenv(dotenv_path)
    return os.getenv(env_var_name)

class ColorLogger:
    COLORS = {
        "RESET": "\033[0m",
        "INFO": "\033[94m",     
        "WARNING": "\033[93m",  
        "ERROR": "\033[91m",   
        "SUCCESS": "\033[92m", 
        "DEBUG": "\033[95m",    
    }

    def __init__(self, enable_debug: bool):
        self.enable_debug = enable_debug

    def _log(self, message: str, level: str):
        if self.enable_debug:
            color = self.COLORS.get(level.upper(), self.COLORS["RESET"])
            print(f"{color}[{level.upper()}] {message}{self.COLORS['RESET']}")

    def info(self, message: str):
        if self.enable_debug:
            self._log(message, "INFO")

    def warning(self, message: str):
        if self.enable_debug:
            self._log(message, "WARNING")

    def error(self, message: str):
        if self.enable_debug:
            self._log(message, "ERROR")

    def success(self, message: str):
        if self.enable_debug:
            self._log(message, "SUCCESS")

    def debug(self, message: str):
        if self.enable_debug:
            self._log(message, "DEBUG")
