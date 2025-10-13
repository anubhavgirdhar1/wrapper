from wrapper.core import Wrapper

# Initialize OpenRouter client
client = Wrapper("openrouter")

# Generate text
response = client.generate(
    model="openai/gpt-4o",
    user="Explain quantum entanglement in simple terms.",
    system="You are a concise and clear physics tutor."
)

print("\nOpenRouter Response:\n", response)
