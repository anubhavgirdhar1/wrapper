# list_models.py

from wrapper.core import Wrapper

def print_models_for_provider(provider_name: str):
    """
    Initializes a provider via the Wrapper and prints its available models.
    """
    print("-" * 50)
    print(f"Fetching models for provider: {provider_name.upper()}")
    print("-" * 50)
    
    try:
        # The available_models_api is a static method that handles
        # initialization and printing the formatted list.
        models = Wrapper.available_models_api(provider_name)
        
        if not models:
            print(f"No models returned for {provider_name}. This could be an API key issue or network problem.")
        else:
            print(f"\nSuccessfully listed {len(models)} models for {provider_name}.")
            
    except Exception as e:
        print(f"\nAn unexpected error occurred while fetching models for {provider_name}:")
        print(e)
    
    print("\n")

if __name__ == "__main__":
    # Ensure your OPENROUTER_API_KEY and GEMINI_API_KEY are in your .env file
    # print_models_for_provider("openrouter")
    # print_models_for_provider("gemini")
    print_models_for_provider("groq")