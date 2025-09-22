from wrapper.core import Wrapper

# Initialize Anthropic client
anthropic_client = Wrapper("anthropic")

# Generate text
response = anthropic_client.generate(
    model="claude-sonnet-4-0",
    user="Hello, how are you?",
    system="You are a helpful assistant."
)
print(response)