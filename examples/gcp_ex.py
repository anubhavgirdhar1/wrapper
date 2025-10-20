from wrapper import Wrapper

# Initialize GCP Vertex AI provider
gcp_client = Wrapper("gcp", project_id="your-project-id", location="us-central1")

# Non-streaming example
response = gcp_client.generate(
    model="gemini-1.5-pro",
    user="Explain quantum computing in simple terms",
    temperature=0.7,
    max_tokens=500
)
print("Non-streaming response:")
print(response)

# Streaming example
print("\nStreaming response:")
stream_response = gcp_client.generate(
    model="gemini-1.5-flash",
    user="Write a short poem about AI",
    stream=True,
    temperature=0.9
)

for chunk in stream_response:
    print(chunk, end="", flush=True)

# List available models
print("\n\nAvailable models:")
models = Wrapper.available_models_api("gcp", project_id="your-project-id")
