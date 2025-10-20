# wrapper/providers/gcp_provider.py
from typing import List, Dict, Any, Union, Iterator
from wrapper.base import BaseLLM
from wrapper.utils import get_or_request_key, ColorLogger
from wrapper.config import *
import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession
from google.api_core import exceptions as google_exceptions

log = ColorLogger(enable_debug=SHOW_LOGS)

class GCPProvider(BaseLLM):
    def __init__(self, project_id: str = None, location: str = "us-central1"):
        """
        Initialize GCP Vertex AI provider
        
        Args:
            project_id: GCP project ID (will prompt if not provided)
            location: GCP region for Vertex AI (default: us-central1)
        """
        log.debug("[gcp] Initializing GCPProvider")
        
        # Get project ID from parameter or environment
        self.project_id = project_id or get_or_request_key(
            "GCP_PROJECT_ID", 
            "Please enter your GCP Project ID"
        )
        
        # Get location from environment or use default
        self.location = location
        
        # Initialize Vertex AI
        try:
            vertexai.init(project=self.project_id, location=self.location)
            log.success(f"[gcp] GCPProvider initialized successfully (project: {self.project_id}, location: {self.location})")
        except Exception as e:
            log.error(f"[gcp] Failed to initialize Vertex AI: {e}")
            raise RuntimeError(f"GCP initialization failed: {str(e)}")
        
        self.default_system_prompt = "You are a helpful AI assistant."
        
        # Common Vertex AI models
        self.available_models = [
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.0-pro",
            "gemini-pro",
            "gemini-pro-vision",
        ]

    def generate(
        self,
        model: str,
        prompt: str = None,
        messages: List[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.95,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Iterator[str]]:
        """
        Generate text using GCP Vertex AI models
        
        Args:
            model: Model name (e.g., 'gemini-1.5-pro')
            prompt: Single prompt string
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling threshold
            stream: Enable streaming responses
            **kwargs: Additional model parameters
        
        Returns:
            Generated text or stream iterator
        """
        log.info(f"[gcp] Starting generation with model: {model}")
        log.debug(f"[gcp] Stream mode: {stream}, Temperature: {temperature}, Max tokens: {max_tokens}")
        
        try:
            # Initialize the model
            generative_model = GenerativeModel(model)
            
            # Prepare the prompt/messages
            if messages and isinstance(messages, list):
                log.debug(f"[gcp] Using provided messages (count: {len(messages)})")
                # Convert messages to Vertex AI format
                prompt_text = self._messages_to_prompt(messages)
            elif prompt:
                log.debug(f"[gcp] Using single prompt: {prompt[:50]}...")
                prompt_text = prompt
            else:
                log.warning("[gcp] No prompt or messages provided, using default")
                prompt_text = "Hello"
            
            # Prepare generation config
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": top_p,
            }
            
            # Add any additional kwargs to config
            for key, value in kwargs.items():
                if value is not None and key not in generation_config:
                    generation_config[key] = value
            
            log.debug(f"[gcp] Generation config: {generation_config}")
            
            if stream:
                log.info("[gcp] Streaming response initiated")
                return self._generate_stream(generative_model, prompt_text, generation_config)
            else:
                log.debug("[gcp] Sending request to Vertex AI")
                response = generative_model.generate_content(
                    prompt_text,
                    generation_config=generation_config
                )
                
                result = response.text.strip()
                log.success(f"[gcp] Generation completed successfully (length: {len(result)} chars)")
                return result
                
        except google_exceptions.PermissionDenied as e:
            log.error("[gcp] Permission denied - check your GCP credentials and API access")
            log.debug(f"[gcp] PermissionDenied details: {e}")
            raise RuntimeError(f"Permission denied: {str(e)}")
        except google_exceptions.NotFound as e:
            log.error(f"[gcp] Model '{model}' not found or not available in region '{self.location}'")
            log.debug(f"[gcp] NotFound details: {e}")
            raise RuntimeError(f"Model not found: {str(e)}")
        except google_exceptions.ResourceExhausted as e:
            log.error("[gcp] Quota exceeded - check your GCP quotas and limits")
            log.debug(f"[gcp] ResourceExhausted details: {e}")
            raise RuntimeError(f"Quota exceeded: {str(e)}")
        except google_exceptions.InvalidArgument as e:
            log.error("[gcp] Invalid request parameters")
            log.debug(f"[gcp] InvalidArgument details: {e}")
            raise RuntimeError(f"Invalid argument: {str(e)}")
        except Exception as e:
            log.error("[gcp] Unexpected error occurred")
            log.debug(f"[gcp] Exception details: {type(e).__name__} - {e}")
            raise RuntimeError(f"Generation failed: {str(e)}")
    
    def _generate_stream(
        self, 
        model: GenerativeModel, 
        prompt: str, 
        generation_config: dict
    ) -> Iterator[str]:
        """
        Generate streaming response
        
        Args:
            model: Initialized GenerativeModel instance
            prompt: Prompt text
            generation_config: Generation configuration dict
            
        Yields:
            Text chunks as they arrive
        """
        try:
            chunk_count = 0
            response_stream = model.generate_content(
                prompt,
                generation_config=generation_config,
                stream=True
            )
            
            for chunk in response_stream:
                if chunk.text:
                    chunk_count += 1
                    yield chunk.text
            
            log.debug(f"[gcp] Streaming completed with {chunk_count} chunks")
            
        except Exception as e:
            log.error(f"[gcp] Streaming error: {type(e).__name__}")
            log.debug(f"[gcp] Streaming error details: {e}")
            raise RuntimeError(f"Streaming failed: {str(e)}")
    
    def _messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """
        Convert messages list to a single prompt string
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            Combined prompt string
        """
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts)
    
    def list_models(self) -> List[str]:
        """
        List available Vertex AI models
        
        Returns:
            List of model IDs
        """
        log.info("[gcp] Listing available Vertex AI models")
        
        try:
            # Return the list of commonly available models
            # Vertex AI doesn't have a simple list API, so we return known models
            log.success(f"[gcp] Successfully retrieved {len(self.available_models)} models")
            log.debug(f"[gcp] Available models: {', '.join(self.available_models)}")
            return self.available_models
            
        except Exception as e:
            log.error("[gcp] Error listing models")
            log.debug(f"[gcp] Exception: {type(e).__name__} - {e}")
            return []
