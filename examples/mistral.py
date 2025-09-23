from wrapper.core import Wrapper

# Initialize Mistral client
mistral_client = Wrapper("mistral")

# List available models
models = mistral_client.available_models_api("mistral")
print("Available models:", models)

# Generate text (non-streaming)
response = mistral_client.generate(
    model="mistral-tiny",  # Replace with a valid model from the list
    prompt="Hello, how are you?",
    system="You are a helpful assistant."
)
print("\nNon-streaming response:\n", response)

# Generate text (streaming)
print("\nStreaming response:")
for chunk in mistral_client.generate(
    model="mistral-tiny",  # Replace with a valid model from the list
    prompt="Tell me a joke.",
    stream=True
):
    print(chunk, end="", flush=True)
