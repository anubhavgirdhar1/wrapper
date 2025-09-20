# TODO Add meaningful Logs using ColorLogger

from ..utils import ColorLogger
from ..base import BaseLLM
from ..utils import get_or_request_key
from groq import Groq
import sys

class GroqProvider(BaseLLM):
    def __init__(self):
        self.api_key = get_or_request_key("GROQ_API_KEY", "Please enter your Groq API Key")
        self.client = Groq(api_key=self.api_key)
    
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
                **kwargs
            )
            return response.choices[0].message.content.strip()
        else:
            output = []
            stream_resp = self.client.chat.completions.create(
                model=model,
                messages=final_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=True, 
                **kwargs
            )
            for chunk in stream_resp:
                delta = chunk.choices[0].delta.content
                if delta:
                    sys.stdout.write(delta)
                    sys.stdout.flush()
                    output.append(delta)
            return "".join(output).strip()
        
    def list_models(self):
        models = self.client.models.list()
        return [m.id for m in models.data]
    