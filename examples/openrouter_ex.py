from wrapper.core import Wrapper
#Initialize OpenRouter client
openrouter_client = Wrapper("openrouter")

response = openrouter_client.generate(
    model="openai/gpt-4o",
    user="Write a tiny poem about summer rain.",
    system="You are a helpful assistant.",
)
print(response)

#Streaming example
stream_response = openrouter_client.generate(
    model="openai/gpt-4o",
    user="Write a tiny poem about summer rain.",
    system="You are a helpful assistant.",
    stream=True
)
for chunk in stream_response:
    print(chunk, end='', flush=True)