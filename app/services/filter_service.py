import yaml
import json
import httpx
import os
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.core.logging import get_logger

logger = get_logger()

class FilterService:
    def __init__(self):
        # Load LLM and prompt configuration from the correct section for OpenRouter
        config_path = Path(__file__).parent.parent / "config" / "llm_config.yaml"
        prompt_dir = config_path.parent
        with open(config_path, "r") as f:
            self.llm_config = yaml.safe_load(f).get("final_ranking_openrouter", {})

        prompt_file = self.llm_config.get("prompt_file", "prompts/final_ranking.txt")
        prompt_file_path = prompt_dir.parent / "config" / prompt_file

        with open(prompt_file_path, "r") as f:
            self.prompt_template = f.read()

        self.model_name = self.llm_config.get("model", "google/gemini-flash-1.5-pro-latest")
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

        # Setup an async HTTP client
        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        logger.info(f"FilterService initialized for OpenRouter. Model: {self.model_name}")

    async def _encode_image_to_base64(self, image_path: Path) -> Optional[str]:
        """Encodes an image file to a base64 string."""
        try:
            if not image_path.exists():
                logger.error(f"Image file not found at {image_path}")
                return None
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {image_path} to base64: {e}", exc_info=True)
            return None

    async def rank_products_with_openrouter(
        self,
        products: List[Dict[str, Any]],
        ground_truth_image_path: Path
    ) -> List[str]:
        """
        Uses an OpenRouter multimodal model to rank products based on a ground truth image.

        Args:
            products: A list of candidate products (deduplicated).
            ground_truth_image_path: The file path to the main video frame for visual comparison.

        Returns:
            A list of 'random_key's, sorted by the model's ranking.
        """
        if not self.api_key:
            logger.error("OPENROUTER_API_KEY not set. Cannot proceed with ranking.")
            return []

        if not products:
            logger.warning("No products provided to rank.")
            return []

        base64_image = await self._encode_image_to_base64(ground_truth_image_path)
        if not base64_image:
            logger.error("Failed to encode ground truth image. Aborting ranking.")
            return []

        # Prepare the product list for the prompt
        product_list_for_prompt = []
        for p in products:
            product_data = {
                "name": p.get("name"),
                "random_key": p.get("random_key"),
                "image_url": p.get("image_url")
            }
            product_list_for_prompt.append(product_data)
        product_list_json = json.dumps(product_list_for_prompt, indent=2, ensure_ascii=False)

        prompt = self.prompt_template.format(
            product_list_json=product_list_json
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "stream": False
        }

        try:
            logger.info(f"Sending request to OpenRouter for final ranking. Products count: {len(products)}")
            response = await self.client.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()

            response_data = response.json()
            response_text = response_data['choices'][0]['message']['content'].strip()
            logger.info(f"Raw OpenRouter ranking response: {response_text}")

            parsed_json = json.loads(response_text)
            ranked_keys = parsed_json.get("ranked_product_keys", [])

            if isinstance(ranked_keys, list) and all(isinstance(k, str) for k in ranked_keys):
                logger.info(f"OpenRouter ranked {len(ranked_keys)} products successfully.")
                return ranked_keys
            else:
                logger.warning(f"OpenRouter response key 'ranked_product_keys' was not a list of strings: {ranked_keys}")
                return []

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred while calling OpenRouter: {e.response.status_code} {e.response.text}", exc_info=True)
            return []
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed for OpenRouter response: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred during OpenRouter ranking: {e}", exc_info=True)
            return []

# Singleton instance
filter_service = FilterService()
