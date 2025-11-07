import httpx
import yaml
from typing import Dict, Any, List

from app.core.config import settings

class EdenAIService:
    def __init__(self):
        self.api_key = settings.EDEN_API_KEY
        self.api_url = "https://api.edenai.run/v2/llm/chat"
        self._load_config()

    def _load_config(self):
        with open("app/config/llm_config.yaml", "r") as f:
            llm_config = yaml.safe_load(f)
        self.product_search_config = llm_config.get("product_search", {})

    async def analyze_video_frames(self, video_base64: str, prompt: str) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.product_search_config.get("model"),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:video/mp4;base64,{video_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.api_url, json=payload, headers=headers, timeout=120)
                response.raise_for_status()
                # print(str(response.text))
                return response.json()
            except httpx.HTTPStatusError as e:
                # Log the error and re-raise or handle it
                print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
                raise
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                raise
