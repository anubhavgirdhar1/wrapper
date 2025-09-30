![image_alt](https://github.com/anubhavgirdhar1/wrapper/blob/8cd8cd4735cdf63030b513aeb5dd173cb866a56a/wrapper/src/banner.png)

[![GitHub stars](https://img.shields.io/github/stars/anubhavgirdhar1/wrapper?style=social)](https://github.com/anubhavgirdhar1/wrapper/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/anubhavgirdhar1/wrapper?style=social)](https://github.com/anubhavgirdhar1/wrapper/network)
[![PyPI version](https://img.shields.io/pypi/v/wrapper)](https://pypi.org/project/wrapper/)
[![License](https://img.shields.io/github/license/anubhavgirdhar1/wrapper)](LICENSE)

---

## Overview

**Wrapper** is a unified Python library designed to interact with multiple Large Language Model (LLM) providers through a **single consistent interface**. Cuz every provider thought 'lets make our own fkin API format so devs can suffer'    

---

## Installation

```bash
# Install via pip
pip install wrapper
```

or if you dont trust PyPI (fair), clone this repo and slap it into your env:

```bash
git clone https://github.com/anubhavgirdhar1/wrapper.git
cd wrapper
pip install -e .
```

---
## .env setup (don’t be that guy who pastes keys into code)

Make a .env file. Throw your keys in it. Done.

```bash
OPENAI_API_KEY=sk-fuckingspam
OLLAMA_BASE_URL=http://localhost:11434
GROQ_API_KEY=groq-faster-than-your-ex
AWS_ACCESS_KEY_ID=xxxx
AWS_SECRET_ACCESS_KEY=xxxx
AWS_REGION=us-east-1
ANTHROPIC_API_KEY=claude-knows-too-much
```

you can ignore this too, wrapper will ask you for it, don't be a dih, none is free.

---

## ⚡ Quick Start

```python
from wrapper import Wrapper

# Initialize provider
OpenAI_Client = Wrapper("openai")
Ollama_Client = Wrapper("ollama")

# Generate text
response = OpenAI_Client.generate(
    model="gpt-4",
    user="say 'dih' if you are alive",
)
print(response)

# Using Ollama with default system prompt
response2 = Ollama_Client.generate(
    model="gemma3n",
    user="say 'puh' if you are alive"
)
print(response2)
```

---

## 🦜 Features

* **One API Model**: all providers are called using single format, stop fooling around the docs.
* **Streaming Support**: Stream responses when available.
* **Custom Prompts**: Custom param support for crazy shi you might wanna pull
* **Environment Management**: Automatically handle API keys via `.env`.
* **Debug Logging**: Toggle debug output via `SHOW_LOGS` in `config.py`. (This a custom logger, try ts out)

---

## 🛠 Supported Providers

| Provider | Why??                                              |
| -------- | -------------------------------------------------- | 
| OpenAI   | the boring dfault shi                              |                                               
| Ollama   | local models so you can LARP on self-sovereign     | 
| Groq     | fast as fuck                                       | 
| Bedrock  | Amazon's way of charging you more than S3          | 
| Anthropic| claude, polite sybau ah model.                     | 
| Azure    | openai but enterprise daddy issues edition         | 

---

## 💪 Contribution
please learn and use conventional commits 

Fork → Branch → PR.
or just maybe open an issue and scream ts out of it

---

## 📜 License

MIT. Do whatever. If it breaks, you get to keep both pieces. See [LICENSE](LICENSE) for details.

---

## 🌐 Links

* **PyPI**: [https://pypi.org/project/wrapper/](https://pypi.org/project/wrapper/)

---

## 👤 Connect with the Author

* **LinkedIn**: [Anubhav Girdhar](https://www.linkedin.com/in/anubhav-girdhar-b82a27225/)

---
