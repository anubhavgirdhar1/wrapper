from wrapper.core import Wrapper

gemini_client = Wrapper("gemini")

response = gemini_client.generate(
    model="models/gemini-2.0-flash",
    user="Write a tiny poem about summer rain.",
    system="You are a helpful assistant.",
    stream=True
)
