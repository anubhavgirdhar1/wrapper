from wrapper.core import Wrapper

openai_client = Wrapper("openai")

response = openai_client.generate(
    model="gpt-5",
    prompt="say yes if you are active",
    stream=True)