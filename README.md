# Ollama API Wrapper

This is a FastAPI application that provides a simple but high performance interface to the Ollama API.

## Features

* Chat with models
* Text generation
* Health check endpoint
* List all available models

### Usage

**Install the requirements by running** `pip install requirements.txt`

To use this application, simply run `python main.py` and access it at http://localhost:8000.

### Endpoints

* `/health`: Check the service's health
* `/models`: List all available models
* `/chat`: Chat with a model
* `/generate`: Generate text using a model
