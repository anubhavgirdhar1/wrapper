from wrapper.core import Wrapper

openrouter_client = Wrapper("openrouter")

response = openrouter_client.generate(
    model="openai/gpt-4o",
    user="Write a tiny poem about summer rain.",
    system="You are a helpful assistant.",
    stream=True
)
