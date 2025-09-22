from .providers.anthropic_provider import AnthropicProvider
from .providers.openai_provider import OpenAIProvider
from .providers.ollama_provider import OllamaProvider
from .providers.groq_provider import GroqProvider
from .providers.bedrock_provider import BedrockProvider
from collections import defaultdict
from .utils import ColorLogger
from .config import *

log = ColorLogger(enable_debug=SHOW_LOGS)

class Wrapper:
    def __init__(self, provider: str, **kwargs):
        provider = provider.lower()

        if provider == "anthropic":
            self.impl = AnthropicProvider(**kwargs)
        elif provider == "openai":
            self.impl = OpenAIProvider(**kwargs)
        elif provider == "ollama":
            self.impl = OllamaProvider(**kwargs)
        elif provider == "groq":  
            self.impl = GroqProvider(**kwargs)
        elif provider == "bedrock":
            self.impl = BedrockProvider(**kwargs)

        else:
            raise ValueError(f"Provider {provider} not supported yet")
        
    def generate(
        self,
        model: str,
        prompt: str = None,
        user: str = None,
        system: str = None,
        messages: list[dict] = None,
        temperature: float = None,
        max_tokens: float = None,
        top_p: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        stream: bool = False,
        **kwargs
    ) -> str:

        # Prepare messages
        if messages is None:
            messages = []

            # system message
            if system:
                messages.append({"role": "system", "content": system})
            elif hasattr(self.impl, "default_system_prompt"):
                messages.append({"role": "system", "content": self.impl.default_system_prompt})

            # user message
            if user:
                messages.append({"role": "user", "content": user})
            elif prompt:
                messages.append({"role": "user", "content": prompt})

        # Build params to pass to provider
        params = {"model": model, "stream": stream, "messages": messages}
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if top_p is not None:
            params["top_p"] = top_p
        if frequency_penalty is not None:
            params["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            params["presence_penalty"] = presence_penalty
        params.update(kwargs)

        # Pass everything to the provider's generate
        return self.impl.generate(**params)

    @staticmethod
    def available_models_api(provider: str, **kwargs):
        provider = provider.lower()

        if provider == "anthropic":
            instance = AnthropicProvider(**kwargs)
            models = instance.list_models()
            log.info("\n Anthropic Models:\n")
            for i, m in enumerate(models, 1):
                log.info(f" {i:2d}. {m}")
            return models
        
        elif provider == "openai":
            instance = OpenAIProvider(**kwargs)
            models = instance.list_models()
            groups = defaultdict(list)
            for m in models:
                if m.startswith(("gpt-3.5", "gpt-4", "gpt-5", "o1", "o3", "o4")):
                    groups["LLMs"].append(m)
                elif "dall-e" in m or "image" in m:
                    groups["Image Models"].append(m)
                elif "tts" in m or "audio" in m or "whisper" in m:
                    groups["Audio Models"].append(m)
                elif "embedding" in m:
                    groups["Embedding Models"].append(m)
                else:
                    groups["Other"].append(m)

            log.info("\n OpenAI Models (by category):\n")
            for category, items in groups.items():
                log.info(f" {category}:")
                for i, model in enumerate(items, 1):
                    log.info(f"   {i:2d}. {model}")
            return models
        
        elif provider == "ollama":
            instance = OllamaProvider(**kwargs)
            return instance.list_models()
        
        elif provider == "groq":  
            instance = GroqProvider(**kwargs)
            models = instance.list_models()
            log.info("\n Groq Models:\n")
            for i, m in enumerate(models, 1):
                log.info(f"   {i:2d}. {m}")
            return models

        elif provider == "bedrock":
            instance = BedrockProvider(**kwargs)
            models = instance.list_models() or []
            log.info("\n Bedrock Models:\n")
            for i, m in enumerate(models, 1):
                log.info(f"   {i:2d}. {m['modelId']} — {m['modelName']} ({m['provider']})")
            return models
        
        log.warning(f"No wrapper available for provider '{provider}'.")
        return []
