from wrapper.core import Wrapper

# Initialize Gemini client
client = Wrapper("gemini")

# Generate text
response = client.generate(
    model="gemini-1.5-flash",  # or "gemini-2.0-flash" if you have access
    user="Summarize the benefits of renewable energy.",
    system="You are an expert in sustainable technologies."
)

print("\nGemini Response:\n", response)
