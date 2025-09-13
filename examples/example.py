from wrapper.core import Wrapper

Ollama_Client = Wrapper("ollama")
# OpenAI_Client = Wrapper("openai")

model = "gemma3n"
user="Just say Yes if you work, nothing else"
system="respond in 2 tokens"
stream = False

# call_type = 'Async' #TODO

# for User, System style calls
print(Ollama_Client.generate(model=model, user=user, system=system , stream=stream))

# for plain call with prompt
# print(Ollama_Client.generate(model=model, prompt=user, stream=stream))

# OpenAI_Client.generate(model=model, prompt=prompt, stream=stream)