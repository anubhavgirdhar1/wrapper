from wrapper.core import Wrapper

client = Wrapper("gemini")

print("\n--- Gemini Streaming ---\n")
resp = client.generate(
    model="gemini-1.5-flash",
    user="Write a tiny poem about summer rain.",
    system="You are creative and poetic.",
    stream=True,
)
print("\n\n[END STREAM]\n", resp)
