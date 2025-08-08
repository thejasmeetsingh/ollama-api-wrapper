"""
This is the main entry point for the Ollama API Wrapper application.
"""

import logging
import asyncio
import multiprocessing
from typing import Annotated, Optional

import uvicorn
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Depends, status
from ollama import AsyncClient, Message, Tool, Options

from pydantic import BaseModel
from pydantic.json_schema import JsonSchemaValue


logger = logging.getLogger(__name__)
app = FastAPI(
    title="Ollama API Wrapper",
    description="A FastAPI application that provides simple but high performance interface to the Ollama API."
)


class BaseSchema(BaseModel):
    """
    Base schema for API requests.
    """

    model: str
    options: Optional[Options] = None
    format: Optional[JsonSchemaValue] = None


class ChatSchema(BaseSchema):
    """
    Schema for chat requests.
    """

    messages: list[Message]
    tools: Optional[list[Tool]] = None


class GenerateSchema(BaseSchema):
    """
    Schema for generate requests.
    """

    prompt: str


def get_async_client():
    """
    Function to create an instance of AsyncClient.

    Yields:
        AsyncClient: An instance of the AsyncClient.
    """

    try:
        client = AsyncClient()
        yield client
    except Exception as e:
        logger.error(f"Failed to create AsyncClient: {e}")


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check(client: Annotated[AsyncClient, Depends(get_async_client)]):
    """
    Endpoint to check the service's health.
    Args:
        client (AsyncClient): An instance of the AsyncClient.
    Returns:
        JSONResponse: A response indicating that the service is up and running.
    """

    await client.list()
    return {"message": "Service is up and running!"}


@app.get("/models", status_code=status.HTTP_200_OK)
async def list_models(client: Annotated[AsyncClient, Depends(get_async_client)]) -> JSONResponse:
    """
    Endpoint to list all available models.

    Args:
        client (AsyncClient): An instance of the AsyncClient.

    Returns:
        JSONResponse: A response containing a list of available models.
    """

    response = await client.list()
    return JSONResponse(content={"models": list(map(lambda model: model.model, response.models))})


@app.post("/chat", status_code=status.HTTP_200_OK)
async def chat(
    client: Annotated[AsyncClient, Depends(get_async_client)],
    request: ChatSchema
) -> JSONResponse:
    """
    Endpoint for chat with the model.

    Args:
        client (AsyncClient): An instance of the AsyncClient.
        model_id (str): The ID of the model to use.
        messages (list[Message]): A list of messages to send to the model.
        tools (list[Tool] | None, optional): A list of tools to use. Defaults to None.
        options (Options | None, optional): Options for the chat. Defaults to None.
        format (JsonSchemaValue | None, optional): The format of the response. Defaults to None.

    Returns:
        JSONResponse: A response containing the result of the chat.
    """

    response = await client.chat(**request.model_dump())
    return JSONResponse(content=response.message.model_dump())


@app.post("/generate", status_code=status.HTTP_200_OK)
async def generate(client: Annotated[AsyncClient, Depends(get_async_client)], request: GenerateSchema) -> JSONResponse:
    """
    Endpoint for text generation with the model.

    Args:
        client (AsyncClient): An instance of the AsyncClient.
        model_id (str): The ID of the model to use.
        prompt (str): The prompt to generate text from.
        options (Options | None, optional): Options for the generation. Defaults to None.
        format (JsonSchemaValue | None, optional): The format of the response. Defaults to None.

    Returns:
        JSONResponse: A response containing the generated text.
    """

    response = await client.generate(**request.model_dump())
    return JSONResponse(content=response.model_dump())


async def main():
    """
    Main entry point for the application.
    """

    workers = multiprocessing.cpu_count() * 2
    config = uvicorn.Config("main:app", port=8000, log_level="info", workers=workers)
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
