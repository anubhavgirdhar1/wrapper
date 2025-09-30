from wrapper.core import Wrapper

bedrock_client = Wrapper('bedrock')

response = bedrock_client.generate(
    model="claude-sonnet-4-0",
    prompt="Hello, how are you?"
)