from wrapper.core import Wrapper
# Initialize Gemini client
gemini_client = Wrapper("gemini")

response = gemini_client.generate(
    model="models/gemini-2.0-flash",
    user="Write a tiny poem about summer rain.",
    system="You are a helpful assistant.",
)
print(response)

# Streaming example
stream_response = gemini_client.generate(
    model="models/gemini-2.0-flash",
    user="Write a tiny poem about summer rain.",
    system="You are a helpful assistant.",
    stream=True
)
for chunk in stream_response:
    print(chunk, end='', flush=True)