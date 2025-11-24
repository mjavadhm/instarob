import yaml
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
import aiofiles
import base64

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger()

class OpenRouterService:
    def __init__(self):
        # Load LLM configuration
        config_path = Path(__file__).parent.parent / "config" / "llm_config.yaml"
        prompt_dir = config_path.parent
        with open(config_path, "r") as f:
            self.llm_config = yaml.safe_load(f).get("text_result_filter_openrouter", {})

        prompt_file = self.llm_config.get("prompt_file", "prompts/text_result_filter.txt")
        prompt_file_path = prompt_dir.parent / "config" / prompt_file
        with open(prompt_file_path, "r") as f:
            self.prompt_template = f.read()

        self.model_name = self.llm_config.get("model", "google/gemini-flash-1.5")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY,
        )
        logger.info(f"OpenRouterService initialized. Model: {self.model_name}")

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
            prompt_tokens = 0
            completion_tokens = 0
            prompt_text = self.prompt_template.format(identified_product=identified_product)

            # Start with the main prompt and the reference video frames
            content = [{"type": "text", "text": prompt_text}]
            for frame_path in frame_paths:
                base64_image = await self._image_to_base64(frame_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })

            # Add a separator text
            content.append({"type": "text", "text": "\n--- PRODUCT LIST ---\nNow, for each product below, decide if it visually matches the reference images above."})

            # For each product, add its details and its image in sequence
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

            logger.info(f"Sending request to OpenRouter with {len(frame_paths)} frames and {len(products)} products.")

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                extra_headers={
                    "HTTP-Referer": "https://instarob.ai",
                    "X-Title": "Instarob AI",
                },
            )

            response_text = completion.choices[0].message.content.strip()
            logger.info(f"Raw OpenRouter Response: {response_text}")

            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            parsed_json = json.loads(response_text)
            relevant_keys = parsed_json.get("relevant_keys", [])

            prompt_tokens = completion.usage.prompt_tokens
            completion_tokens = completion.usage.completion_tokens

            if isinstance(relevant_keys, list) and all(isinstance(k, str) for k in relevant_keys):
                logger.info(f"OpenRouter filtered down to {len(relevant_keys)} relevant products.")
                return relevant_keys, prompt_tokens, completion_tokens
            else:
                logger.warning(f"OpenRouter response key 'relevant_keys' was not a list of strings: {relevant_keys}")
                return [], prompt_tokens, completion_tokens

        except Exception as e:
            logger.error(f"An error occurred during OpenRouter filtering: {e}", exc_info=True)
            return [], 0, 0

    # ... (existing methods) ...

    async def rank_and_filter_final_list(
        self,
        products: List[Dict[str, Any]],
        frame_paths: List[Path],
        identified_product: str
    ) -> Tuple[List[str], int, int, str]:
        try:
            # Load config for this specific method
            config_path = Path(__file__).parent.parent / "config" / "llm_config.yaml"
            with open(config_path, "r") as f:
                final_rank_config = yaml.safe_load(f).get("final_ranking_openrouter", {})

            prompt_file = final_rank_config.get("prompt_file", "prompts/final_ranking.txt")
            prompt_file_path = config_path.parent / prompt_file
            with open(prompt_file_path, "r") as f:
                prompt_template = f.read()

            model_name = final_rank_config.get("model", self.model_name)

            prompt_text = prompt_template.format(identified_product=identified_product)

            content = [{"type": "text", "text": prompt_text}]
            for frame_path in frame_paths:
                base64_image = await self._image_to_base64(frame_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })

            content.append({"type": "text", "text": "\n--- PRODUCT LIST TO RANK ---\n"})

            for i, product in enumerate(products):
                # Ensure product image URL exists
                image_url = product.get("image_url")
                if not image_url and product.get("source") == "image":
                    # This is a bit of a hack: reconstruct image URL if not present for image-sourced products
                    # This depends on knowing the structure of how image search results are stored/processed
                    # A better solution would be to ensure image_url is always present
                    pass # Or find a way to get the URL

                product_info = (
                    f"\n\nProduct {i+1}:\n"
                    f"Name: {product.get('name')}\n"
                    f"Random Key: {product.get('random_key')}"
                )
                content.append({"type": "text", "text": product_info})

                if image_url:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    })

            messages = [{"role": "user", "content": content}]

            completion = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                extra_headers={
                    "HTTP-Referer": "https://instarob.ai",
                    "X-Title": "Instarob AI",
                },
            )

            response_text = completion.choices[0].message.content.strip()
            logger.info(f"Raw OpenRouter Final Ranking Response: {response_text}")

            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            parsed_json = json.loads(response_text)
            ranked_keys = parsed_json.get("ranked_keys", [])

            prompt_tokens = completion.usage.prompt_tokens
            completion_tokens = completion.usage.completion_tokens

            if isinstance(ranked_keys, list) and all(isinstance(k, str) for k in ranked_keys):
                logger.info(f"OpenRouter ranked and filtered down to {len(ranked_keys)} products.")
                return ranked_keys, prompt_tokens, completion_tokens, model_name
            else:
                logger.warning(f"OpenRouter response key 'ranked_keys' was not a list of strings: {ranked_keys}")
                return [], prompt_tokens, completion_tokens, model_name

        except Exception as e:
            logger.error(f"An error occurred during OpenRouter final ranking: {e}", exc_info=True)
            return [], 0, 0, ""

openrouter_service = OpenRouterService()
