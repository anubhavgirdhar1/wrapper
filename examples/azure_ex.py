from wrapper.core import Wrapper

# Initialize Azure client
azure_client = Wrapper("azure")

# Generate text
response = azure_client.generate(
    model="gpt-4",
    user="Hello, how are you?",
    system="You are a helpful assistant."
)