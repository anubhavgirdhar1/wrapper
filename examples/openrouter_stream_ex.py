from wrapper.core import Wrapper

client = Wrapper("openrouter")

print("\n--- OpenRouter Streaming ---\n")
resp = client.generate(
    model="openai/gpt-4o",
    user="Tell me a short, funny joke about databases.",
    system="You are witty and brief.",
    stream=True,
)
print("\n\n[END STREAM]\n", resp)
