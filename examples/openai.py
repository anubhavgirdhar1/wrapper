from wrapper.core import Wrapper

openai_client = Wrapper("openai")

response = openai_client.generate(
    model="gpt-4",
    prompt="say yes if you are active",
    stream=True)