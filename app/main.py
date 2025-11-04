import google.generativeai as genai
from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import settings
from app.core.logging import setup_logging, RequestContextLogMiddleware

# Configure the Gemini client on application startup
if settings.GEMINI_API_KEY:
    genai.configure(api_key=settings.GEMINI_API_KEY)

setup_logging()

app = FastAPI(title="Reels Input Service")
app.add_middleware(RequestContextLogMiddleware)
app.include_router(api_router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Reels Input Service is running"}
