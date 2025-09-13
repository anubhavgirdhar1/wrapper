import sys
from pathlib import Path
from openai import OpenAI, AuthenticationError
from ..base import BaseLLM
from ..utils import get_or_request_key, ColorLogger
from ..config import *

log = ColorLogger(enable_debug=SHOW_LOGS)

class OpenAIProvider(BaseLLM):
    def __init__(self):
        # auto fetch API key from env or ask user
        self.api_key = get_or_request_key("OPENAI_API_KEY", "Please enter your OpenAI API Key")
        self.client = OpenAI(api_key=self.api_key)

    def generate(
        self, 
        model: str,
        prompt: str = None,
        messages: list[dict] = None,
        temperature: float = 0.7, 
        max_tokens: int = 200, 
        top_p: float = 0.1, 
        frequency_penalty: int = 0, 
        presence_penalty: int = 0, 
        stream: bool = False, 
        **kwargs
    ) -> str:
        # Decide between messages and prompt
        if messages:
            final_messages = messages
        else:
            # Default system + user
            system_default = {"role": "system", "content": "You are a helpful AI assistant."}
            user_msg = {"role": "user", "content": prompt}
            final_messages = [system_default, user_msg]

        if not stream:
            response = self.client.chat.completions.create(
                model=model,
                messages=final_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                **kwargs
            )
            return response.choices[0].message.content.strip()
        else:
            output = []
            with self.client.chat.completions.stream(
                model=model,
                messages=final_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                **kwargs
            ) as stream_resp:
                for event in stream_resp:
                    if hasattr(event, "type") and event.type == "content.delta":
                        delta_content = getattr(event, "delta", "")
                        if delta_content:
                            sys.stdout.write(delta_content)
                            sys.stdout.flush()
                            output.append(delta_content)
            return "".join(output).strip()

    def list_models(self) -> list[str]:
        try:
            models = self.client.models.list()
            return sorted([m.id for m in models.data])
        except AuthenticationError:
            raise RuntimeError("Invalid API key. Please check .env file.")
