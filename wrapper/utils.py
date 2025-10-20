import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from functools import wraps
from typing import Callable, Any, Type, Tuple

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


def with_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retriable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    logger: 'ColorLogger' = None
):
    """
    Decorator to add retry logic with exponential backoff to any function.
    
    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay between retries in seconds (default: 1.0)
        backoff_factor: Multiplier for delay after each retry (default: 2.0)
        retriable_exceptions: Tuple of exception types that should trigger retry
        logger: Optional ColorLogger instance for logging retry attempts
        
    Returns:
        Decorated function with retry logic
        
    Example:
        @with_retry(max_retries=3, initial_delay=1.0, backoff_factor=2.0)
        def call_api():
            # API call that might fail
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except retriable_exceptions as e:
                    last_exception = e
                    
                    # Don't retry on last attempt
                    if attempt >= max_retries:
                        if logger:
                            logger.error(f"[retry] Max retries ({max_retries}) exceeded for {func.__name__}")
                        raise
                    
                    # Log retry attempt
                    if logger:
                        logger.warning(
                            f"[retry] Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: "
                            f"{type(e).__name__} - {str(e)[:100]}"
                        )
                        logger.info(f"[retry] Retrying in {delay:.1f} seconds...")
                    
                    # Wait before retrying
                    time.sleep(delay)
                    delay *= backoff_factor
                    
                except Exception as e:
                    # Non-retriable exception, raise immediately
                    if logger:
                        logger.error(f"[retry] Non-retriable exception in {func.__name__}: {type(e).__name__}")
                    raise
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


class ColorLogger:
    COLORS = {
        "RESET": "\033[0m",
        "INFO": "\033[94m",     
        "WARNING": "\033[93m",  
        "ERROR": "\033[91m",   
        "SUCCESS": "\033[92m", 
        "DEBUG": "\033[95m",    
    }

    ASCII_REMINDER = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║  PSST! These debug logs annoying you?                         ║
    ║  Go to your config.py and set:                                ║
    ║                                                               ║
    ║      SHOW_LOGS = False                                        ║
    ║                                                               ║
    ║  This will shut me up and make your terminal clean again      ║
    ╚═══════════════════════════════════════════════════════════════╝
    """

    def __init__(self, enable_debug: bool):
        self.enable_debug = enable_debug
        self._shown_reminder = False

    def _log(self, message: str, level: str):
        if self.enable_debug:
            color = self.COLORS.get(level.upper(), self.COLORS["RESET"])
            print(f"{color}[{level.upper()}] {message}{self.COLORS['RESET']}")

    def _show_reminder_once(self):
        """Show the ASCII reminder only once per session"""
        if self.enable_debug and not self._shown_reminder:
            print(self.COLORS["WARNING"] + self.ASCII_REMINDER + self.COLORS["RESET"])
            self._shown_reminder = True

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
            self._show_reminder_once() 