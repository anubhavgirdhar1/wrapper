from setuptools import setup, find_packages

setup(
    name="wrapper", 
    version="0.1.11",
    author="Anubhav Girdhar",
    author_email="anubhavgirdhar18@gmail.com",
    description="A unified Python wrapper for calling LLMs locally or via a 3rd party provider",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/anubhavgirdhar1/wrapper",
    packages=find_packages(),
    include_package_data=True, 
    install_requires=[
        "openai",
        "python-dotenv",
        "requests==2.32.5",
        "anthropic",
        "boto3",
        "botocore",
        "groq-sdk"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
    ],
    python_requires='>=3.8',
    license="Apache License 2.0",
)
