import os
import logging

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import Response
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import httpx

from starlette.responses import JSONResponse

from openai import OpenAI

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def create_httpx_client():
    """Creates an httpx client with desired configurations."""
    return httpx.AsyncClient(
        follow_redirects=True,
        timeout=60.0,
    )


class AIRequest(BaseModel):
    prompt: str


open_ai = OpenAI(os.getenv("OPENAIKEY"))


@app.post("/generate")
async def generate_text(request: AIRequest):
    """
    Generates text using the Open AI model based on the provided prompt.
    """

    prompt: str = request.prompt
    logging.info(f"Received prompt: {prompt}")
    response: str = open_ai.get_response(prompt)
    return JSONResponse(content={"response": response})


@app.get("/{serviceid:str}/")
@app.get("/{serviceid:str}/{path:path}")
async def get_proxy(serviceid: str, path: str = None, request: Request= None, client: httpx.AsyncClient = Depends(create_httpx_client)):
    """Proxies GET requests to the Google Apps Script web app."""

    if path is None:
        target_url = f"https://script.google.com/macros/s/{serviceid}/exec"
    else:
        target_url = f"https://script.google.com/macros/s/{serviceid}/exec?query=/{path}"

    query_params = request.query_params

    if query_params:
        target_url += "?" + str(query_params)  # string conversion required

    logging.info(f"GET Proxying to: {target_url}")

    try:
        response = await client.get(target_url)
        logging.info(response.text)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()

    except httpx.HTTPStatusError as e:
        logging.error(f"HTTP error: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except httpx.RequestError as e:
        logging.error(f"Request error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logging.exception(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")



if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
