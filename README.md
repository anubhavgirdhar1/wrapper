![image_alt](https://github.com/anubhavgirdhar1/wrapper/blob/8cd8cd4735cdf63030b513aeb5dd173cb866a56a/wrapper/src/banner.png)

[![GitHub stars](https://img.shields.io/github/stars/anubhavgirdhar1/wrapper?style=social)](https://github.com/anubhavgirdhar1/wrapper/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/anubhavgirdhar1/wrapper?style=social)](https://github.com/anubhavgirdhar1/wrapper/network)
[![PyPI version](https://img.shields.io/pypi/v/wrapper)](https://pypi.org/project/wrapper/)
[![License](https://img.shields.io/github/license/anubhavgirdhar1/wrapper)](LICENSE)

---

## Overview

**Wrapper** is a unified Python library designed to interact with multiple Large Language Model (LLM) providers through a **single consistent interface**.
It currently supports:

* **OpenAI** (Text based GPT Models)
* **Ollama** (Text based local models)
* **Groq** (Production Models from Groq)
* **Bedrock** (Text Based Models via Bedrock from Anthropic)
* **Anthropic** (Text Based Models directly via Anthropic)
* **Azure AI** (Text Based Models via Azure AI from OpenAI)


The library handles:

* API key management with `.env` integration
* Default and custom system/user prompts
* Streaming outputs (optional, provider-specific)
* Model listing and categorization
* Clean and consistent logging with debug mode

This library is ideal for developers building **multi-LLM applications** without worrying about provider-specific APIs.

---

## 📦 Installation

```bash
# Install via pip
pip install wrapper
```

Or clone the repo:

```bash
git clone https://github.com/anubhavgirdhar1/wrapper.git
cd wrapper
pip install -e .
```

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
    user="say yes if you are active",
)
print(response)

# Using Ollama with default system prompt
response2 = Ollama_Client.generate(
    model="gemma3n",
    user="say yes if you are active"
)
print(response2)
```

---

## 🦜 Features

* **Unified Interface**: Same `generate` and `list_models` for all supported providers.
* **Streaming Support**: Stream responses when available.
* **Custom Prompts**: Pass `system` and `user` messages for fine-grained control.
* **Environment Management**: Automatically handle API keys via `.env`.
* **Debug Logging**: Toggle debug output via `SHOW_LOGS` in `config.py`.

---

## 🛠 Supported Providers

| Provider | Models Supported                                   | Notes                                         |
| -------- | -------------------------------------------------- | ----------------------------------------------|
| OpenAI   | Text Based LLMs | Requires `OPENAI_API_KEY`        |                                               |
| Ollama   | Gemma series (local models)                        | Local endpoint, optional `OLLAMA_API_KEY`     |
| Groq     | Text Based LLMs | Requires `Groq API Key`          | Only Prod. Approved Models are allowed        |
| Bedrock  | Text Based LLMs | Requires account creds.          | Only Text Based models | Preferably Anthropic |
---

## 📜 Advanced Usage

### Custom System Prompts

```python
response = OpenAI_Client.generate(
    model="gpt-4",
    user="Explain recursion in simple terms.",
    system="You are a friendly and concise AI tutor."
)
print(response)
```

### Listing Models

```python
# Just put provider name - for example Ollama

ollama_models = Wrapper.available_models_wrapper("ollama")

```

---

## ⚙ Configuration

Control debug output via `config.py`:

```python
SHOW_LOGS = True  # or False
```

API keys are automatically loaded from `.env` or prompted:

```
OPENAI_API_KEY=your_openai_key
OLLAMA_API_KEY=your_ollama_key
OLLAMA_BASE_URL=http://localhost:11434
```

---

## 🎨 Logging

`ColorLogger` is used across the library to show colored messages:
Logs can be controlled by setting SHOW_LOGS = True in config.py

```python
from utils import ColorLogger
from config import SHOW_LOGS

log = ColorLogger(enable_debug=SHOW_LOGS)
log.info("Info message")
log.warning("Warning message")
log.success("Success message")
log.debug("Debug message (only if SHOW_LOGS=True)")
```

---

## 💪 Contribution

We welcome contributions!

1. Fork the repository
2. Create a new branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -am 'Add feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🌐 Links

* **PyPI**: [https://pypi.org/project/wrapper/](https://pypi.org/project/wrapper/)

---

## 👤 Connect with the Author

* **LinkedIn**: [Anubhav Girdhar](https://www.linkedin.com/in/anubhav-girdhar-b82a27225/)

---
