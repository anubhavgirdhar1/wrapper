from wrapper.core import Wrapper

bedrock_client = Wrapper('bedrock')

response = bedrock_client.generate(
    model="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    prompt="Hello, how are you?"
)