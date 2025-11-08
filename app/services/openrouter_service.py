import yaml
import json
import httpx
import aiofiles
import base64
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from openai import AsyncOpenAI
from pydantic import ValidationError

from app.core.config import settings
from app.core.logging import get_logger
from app.core.utils import async_retry
from app.models.caption_analysis import CaptionAnalysis

logger = get_logger()

class OpenRouterService:
    def __init__(self):
        config_path = Path(__file__).parent.parent / "config" / "llm_config.yaml"
        prompt_dir = config_path.parent
        with open(config_path, "r") as f:
            self.llm_configs = yaml.safe_load(f)

        # Config for Text Result Filter
        text_filter_config = self.llm_configs.get("text_result_filter_openrouter", {})
        self.text_filter_model = text_filter_config.get("model", "google/gemini-flash-1.5")
        prompt_file_path = prompt_dir.parent / "config" / text_filter_config.get("prompt_file")
        with open(prompt_file_path, "r") as f:
            self.text_filter_prompt_template = f.read()

        # Config for Image Search Filter
        image_filter_config = self.llm_configs.get("image_search_filter", {})
        self.image_filter_model = image_filter_config.get("model", "google/gemini-pro-vision")
        prompt_file_path = prompt_dir.parent / "config" / image_filter_config.get("prompt_file")
        with open(prompt_file_path, "r") as f:
            self.image_filter_prompt_template = f.read()

        # Config for Caption Analysis
        caption_analysis_config = self.llm_configs.get("caption_analysis", {})
        self.caption_analysis_model = caption_analysis_config.get("model", "google/gemini-flash-1.5")
        prompt_file_path = prompt_dir.parent / "config" / caption_analysis_config.get("prompt_file")
        with open(prompt_file_path, "r") as f:
            self.caption_analysis_prompt_template = f.read()

        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY,
        )
        logger.info(f"OpenRouterService initialized with AsyncOpenAI.")

    async def _image_to_base64(self, image_path: Path) -> str:
        async with aiofiles.open(image_path, "rb") as f:
            binary_data = await f.read()
            return base64.b64encode(binary_data).decode('utf-8')

    @async_retry()
    async def analyze_caption_for_search_query(self, caption: str) -> Tuple[Optional[CaptionAnalysis], int, int]:
        try:
            prompt = self.caption_analysis_prompt_template.format(caption=caption)
            messages = [{"role": "user", "content": prompt}]

            logger.info(f"Sending async request to OpenRouter for caption analysis.")

            completion = await self.client.chat.completions.create(
                model=self.caption_analysis_model,
                messages=messages,
                extra_headers={
                    "HTTP-Referer": "https://instarob.ai",
                    "X-Title": "Instarob AI",
                },
                response_format={"type": "json_object"},
            )

            response_text = completion.choices[0].message.content.strip()
            logger.info(f"Raw OpenRouter Response (Caption Analysis): {response_text}")
            response_text = response_text.strip().removeprefix("```json").removesuffix("```")
            parsed_json = json.loads(response_text)
            analysis_result = CaptionAnalysis(**parsed_json)

            prompt_tokens = completion.usage.prompt_tokens
            completion_tokens = completion.usage.completion_tokens

            logger.info(f"Caption analysis successful. Query: '{analysis_result.search_query_persian}', Confidence: {analysis_result.confidence}")
            return analysis_result, prompt_tokens, completion_tokens

        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to parse or validate OpenRouter response for caption analysis: {e}", exc_info=True)
            return None, 0, 0

    @async_retry()
    async def filter_text_search_results(
        self,
        products: List[Dict[str, Any]],
        frame_paths: List[Path],
        identified_product: str
    ) -> Tuple[List[str], int, int]:
        try:
            prompt_text = self.text_filter_prompt_template.format(identified_product=identified_product)
            content = [{"type": "text", "text": prompt_text}]

            for frame_path in frame_paths:
                base64_image = await self._image_to_base64(frame_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })

            content.append({"type": "text", "text": "\n--- PRODUCT LIST ---\nNow, for each product below, decide if it visually matches the reference images above."})

            for i, product in enumerate(products):
                product_info = (
                    f"\n\nProduct {i+1}:\n"
                    f"Name: {product.get('name')}\n"
                    f"Random Key: {product.get('random_key')}"
                )
                content.append({"type": "text", "text": product_info})
                if product.get("image_url"):
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": product.get("image_url")}
                    })

            messages = [{"role": "user", "content": content}]

            logger.info(f"Sending async request to OpenRouter with {len(frame_paths)} frames and {len(products)} products for text filtering.")

            completion = await self.client.chat.completions.create(
                model=self.text_filter_model,
                messages=messages,
                extra_headers={
                    "HTTP-Referer": "https://instarob.ai",
                    "X-Title": "Instarob AI",
                },
                response_format={"type": "json_object"},
            )

            response_text = completion.choices[0].message.content.strip().removeprefix("```json").removesuffix("```").strip()
            logger.info(f"Raw OpenRouter Response (Text Filter): {response_text}")

            parsed_json = json.loads(response_text)
            relevant_keys = parsed_json.get("relevant_keys", [])

            prompt_tokens = completion.usage.prompt_tokens
            completion_tokens = completion.usage.completion_tokens

            if isinstance(relevant_keys, list) and all(isinstance(k, str) for k in relevant_keys):
                logger.info(f"OpenRouter filtered down to {len(relevant_keys)} relevant products from text search.")
                return relevant_keys, prompt_tokens, completion_tokens
            else:
                logger.warning(f"OpenRouter response key 'relevant_keys' was not a list of strings: {relevant_keys}")
                return [], prompt_tokens, completion_tokens

        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to parse or validate OpenRouter response for text filtering: {e}", exc_info=True)
            return [], 0, 0

    @async_retry()
    async def filter_image_search_results(
        self,
        products: List[Dict[str, Any]],
        frame_paths: List[Path],
        identified_product: str,
        product_description: str
    ) -> Tuple[List[str], int, int]:
        try:
            prompt_text = self.image_filter_prompt_template.format(
                identified_product=identified_product,
                product_description=product_description
            )
            content = [{"type": "text", "text": prompt_text}]

            for frame_path in frame_paths:
                base64_image = await self._image_to_base64(frame_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })

            content.append({"type": "text", "text": "\n--- CANDIDATE PRODUCT LIST ---\n"})

            for i, product in enumerate(products):
                product_info = (
                    f"\n\nCandidate {i+1}:\n"
                    f"Name: {product.get('title')}\n" # Note: Key is 'title' from image search results
                    f"Random Key: {product.get('random_key')}"
                )
                content.append({"type": "text", "text": product_info})
                if product.get("image_url"):
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": product.get("image_url")}
                    })

            messages = [{"role": "user", "content": content}]

            logger.info(f"Sending async request to OpenRouter with {len(frame_paths)} frames and {len(products)} products for image filtering.")

            completion = await self.client.chat.completions.create(
                model=self.image_filter_model,
                messages=messages,
                extra_headers={
                    "HTTP-Referer": "https://instarob.ai",
                    "X-Title": "Instarob AI",
                },
                response_format={"type": "json_object"},
            )

            response_text = completion.choices[0].message.content.strip().removeprefix("```json").removesuffix("```").strip()
            logger.info(f"Raw OpenRouter Response (Image Filter): {response_text}")

            parsed_json = json.loads(response_text)
            ranked_products = parsed_json.get("ranked_products", [])

            prompt_tokens = completion.usage.prompt_tokens
            completion_tokens = completion.usage.completion_tokens

            if isinstance(ranked_products, list):
                ranked_keys = [p.get("key") for p in ranked_products if p.get("key")]
                logger.info(f"OpenRouter filtered and ranked {len(ranked_keys)} products from image search.")
                return ranked_keys, prompt_tokens, completion_tokens
            else:
                logger.warning(f"OpenRouter response key 'ranked_products' was not a list of objects: {ranked_products}")
                return [], prompt_tokens, completion_tokens

        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to parse or validate OpenRouter response for image filtering: {e}", exc_info=True)
            return [], 0, 0


openrouter_service = OpenRouterService()
