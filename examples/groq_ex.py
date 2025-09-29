from wrapper.core import Wrapper

groq_client = Wrapper("groq")

response = groq_client.generate(
    model="llama-3.3-70b-versatile"
)