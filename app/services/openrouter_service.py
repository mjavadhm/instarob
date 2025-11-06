import yaml
import json
import httpx
import aiofiles
import base64
from pathlib import Path
from typing import List, Dict, Any, Tuple
from openai import AsyncOpenAI

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger()

class OpenRouterService:
    def __init__(self):
        config_path = Path(__file__).parent.parent / "config" / "llm_config.yaml"
        prompt_dir = config_path.parent.parent / "prompts"
        with open(config_path, "r") as f:
            llm_configs = yaml.safe_load(f)

        # Config for Text Result Filtering
        self.text_filter_config = llm_configs.get("text_result_filter_openrouter", {})
        self.text_filter_model = self.text_filter_config.get("model")
        text_filter_prompt_file = prompt_dir / self.text_filter_config.get("prompt_file")
        with open(text_filter_prompt_file, "r") as f:
            self.text_filter_prompt_template = f.read()

        # Config for Caption Analysis
        self.caption_analysis_config = llm_configs.get("caption_analysis", {})
        self.caption_analysis_model = self.caption_analysis_config.get("model")
        caption_analysis_prompt_file = prompt_dir / self.caption_analysis_config.get("prompt_file")
        with open(caption_analysis_prompt_file, "r") as f:
            self.caption_analysis_prompt_template = f.read()

        # Shared AsyncOpenAI client
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY,
        )
        logger.info("OpenRouterService initialized with multiple configurations.")

    async def analyze_caption(self, caption: str) -> Tuple[str | None, int, int]:
        """Analyzes a caption to extract a Persian search query."""
        try:
            prompt = self.caption_analysis_prompt_template.format(caption=caption)
            messages = [{"role": "user", "content": prompt}]

            logger.info(f"Sending caption analysis request to OpenRouter model: {self.caption_analysis_model}")

            completion = await self.client.chat.completions.create(
                model=self.caption_analysis_model,
                messages=messages,
                extra_headers={
                    "HTTP-Referer": "https://instarob.ai",
                    "X-Title": "Instarob AI - Caption Analysis",
                },
                response_format={"type": "json_object"},
                temperature=self.caption_analysis_config.get("temperature", 0.2),
            )

            response_text = completion.choices[0].message.content.strip()
            logger.info(f"Raw OpenRouter Response (Caption Analysis): {response_text}")

            parsed_json = json.loads(response_text)
            search_query = parsed_json.get("search_query_persian")

            prompt_tokens = completion.usage.prompt_tokens
            completion_tokens = completion.usage.completion_tokens

            if search_query and isinstance(search_query, str):
                logger.info(f"Extracted search query: '{search_query}'")
                return search_query, prompt_tokens, completion_tokens
            else:
                logger.info("No valid search query found in caption.")
                return None, prompt_tokens, completion_tokens

        except Exception as e:
            logger.error(f"An error occurred during caption analysis: {e}", exc_info=True)
            return None, 0, 0

    async def _image_to_base64(self, image_path: Path) -> str:
        async with aiofiles.open(image_path, "rb") as f:
            binary_data = await f.read()
            return base64.b64encode(binary_data).decode('utf-8')

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

            response_text = completion.choices[0].message.content.strip()
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

        except Exception as e:
            logger.error(f"An error occurred during OpenRouter text filtering: {e}", exc_info=True)
            return [], 0, 0

openrouter_service = OpenRouterService()
