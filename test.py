# test_providers.py

import sys
from wrapper.core import Wrapper

def run_test(provider_name: str, model: str):
    """
    Runs a standard and streaming test for a given provider and model.
    """
    print("-" * 50)
    print(f"Testing Provider: {provider_name.upper()}")
    print(f"Model: {model}")
    print("-" * 50)

    try:
        client = Wrapper(provider_name)
    except Exception as e:
        print(f"Failed to initialize client for {provider_name}: {e}")
        return

    user_prompt = "Write a short, 1000 words of essay about a fight between zeus and hulk in 2050 with their full power."
    system_prompt = "You are a writer"

    # --- Non-Streaming Test ---
    print("\n--- Running Non-Streaming Test ---")
    try:
        response = client.generate(
            model=model,
            user=user_prompt,
            system=system_prompt,
            stream=False
        )
        print("Response:")
        print(response)
    except Exception as e:
        print(f"\nError during non-streaming test for {provider_name}: {e}")


    # --- Streaming Test ---
    print("\n--- Running Streaming Test ---")
    try:
        stream = client.generate(
            model=model,
            user=user_prompt,
            system=system_prompt,
            stream=True
        )
        
        print("Streaming Response:")
        # The check for the 'API Error' string is a simple way
        # to see if the provider returned an error message instead of an iterator.
        if isinstance(stream, str) and "API Error" in stream:
            print(stream)
        else:
            for chunk in stream:
                print(chunk, end="", flush=True)
        print("\n") # Newline after the stream finishes.
    except Exception as e:
        print(f"\nError during streaming test for {provider_name}: {e}")

if __name__ == "__main__":
    # --- Configuration ---
    # Note: Using a free, reliable model for OpenRouter.
    # For Gemini, 'flash' is a good, fast choice.
    # Make sure your API keys are in your .env file!
    
    openrouter_model = "mistralai/mistral-7b-instruct-v0.1"
    gemini_model = "models/gemini-2.0-flash"
    groq_model = "openai/gpt-oss-120b"
    # --- Run Tests ---
    # run_test("openrouter", openrouter_model)
    # run_test("gemini", gemini_model)
    run_test("groq", groq_model)
    print("-" * 50)
    print("All tests completed.")
    print("-" * 50)