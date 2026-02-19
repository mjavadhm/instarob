import yaml
import json
import httpx
import asyncio
import base64
import aiofiles
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from openai import AsyncOpenAI

from app.core.config import settings
from app.core.logging import get_logger
from app.core.utils import async_retry

logger = get_logger()

class FilterService:
    def __init__(self):
        config_path = Path(__file__).parent.parent / "config" / "llm_config.yaml"
        prompt_dir = config_path.parent
        with open(config_path, "r") as f:
            self.llm_config = yaml.safe_load(f).get("final_ranking_openrouter", {})

        prompt_file = self.llm_config.get("prompt_file", "prompts/final_ranking.txt")
        prompt_file_path = prompt_dir.parent / "config" / prompt_file
        with open(prompt_file_path, "r") as f:
            self.prompt_template = f.read()

        self.model_name = self.llm_config.get("model", "google/gemini-flash-1.5-pro-latest")

        # Use AsyncOpenAI client
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY,
        )
        logger.info(f"FilterService initialized with AsyncOpenAI for final ranking. Model: {self.model_name}")

    async def _image_to_base64(self, image_path: Path) -> str:
        async with aiofiles.open(image_path, "rb") as f:
            binary_data = await f.read()
            return base64.b64encode(binary_data).decode('utf-8')

    async def _download_and_encode_image(self, url: str) -> Optional[str]:
        """Downloads an image from a URL and returns it as a base64 string."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30)
                response.raise_for_status()
                return base64.b64encode(response.content).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to download or encode image from {url}: {e}", exc_info=True)
            return None

    @async_retry()
    async def rank_products_with_llm(
        self,
        products: List[Dict[str, Any]],
        ground_truth_frame_paths: List[Path],
        identified_product: str
    ) -> Tuple[List[str], int, int, str]:
        try:
            prompt_text = self.prompt_template.format(identified_product=identified_product)

            content = [{"type": "text", "text": prompt_text}]
            for frame_path in ground_truth_frame_paths:
                base64_image = await self._image_to_base64(frame_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })

            content.append({"type": "text", "text": "\n--- PRODUCT LIST TO RANK ---\nHere are the candidate products. Please rank them based on their visual similarity to the product in the reference images."})

            image_urls = [p.get("image_url") for p in products if p.get("image_url")]
            image_coroutines = [self._download_and_encode_image(url) for url in image_urls]
            base64_images = await asyncio.gather(*image_coroutines)
            url_to_base64_map = dict(zip(image_urls, base64_images))

            for i, product in enumerate(products):
                product_info = (
                    f"\n\nProduct {i+1}:\n"
                    f"Name: {product.get('name')}\n"
                    f"Random Key: {product.get('random_key')}"
                )
                content.append({"type": "text", "text": product_info})
                image_url = product.get("image_url")
                if image_url and url_to_base64_map.get(image_url):
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{url_to_base64_map[image_url]}"}
                    })
                else:
                    content.append({"type": "text", "text": "Image not available."})

            messages = [{"role": "user", "content": content}]

            logger.info(f"Sending async request to OpenRouter for final ranking with {len(products)} products.")
            completion = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                extra_headers={
                    "HTTP-Referer": "https://instarob.ai",
                    "X-Title": "Instarob AI",
                },
                response_format={"type": "json_object"},
            )

            response_text = completion.choices[0].message.content.strip()
            logger.info(f"Raw OpenRouter Final Ranking Response: {response_text}")

            parsed_json = json.loads(response_text)
            ranked_keys = parsed_json.get("ranked_keys", [])

            prompt_tokens = completion.usage.prompt_tokens
            completion_tokens = completion.usage.completion_tokens

            if isinstance(ranked_keys, list) and all(isinstance(k, str) for k in ranked_keys):
                logger.info(f"OpenRouter ranked {len(ranked_keys)} products.")
                return ranked_keys, prompt_tokens, completion_tokens, self.model_name
            else:
                logger.warning(f"OpenRouter response key 'ranked_keys' was not a list of strings: {ranked_keys}")
                return [], prompt_tokens, completion_tokens, self.model_name

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response from OpenRouter: {e}", exc_info=True)
            return [], 0, 0, self.model_name

filter_service = FilterService()
