import yaml
import json
from pathlib import Path
from typing import List, Dict, Any
import google.generativeai as genai

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger()

class FilterService:
    def __init__(self):
        # Configure Gemini client, assuming it's already configured in ReelsService
        if not genai.get_api_key():
            genai.configure(api_key=settings.GEMINI_API_KEY)

        # Load LLM and prompt configuration
        config_path = Path(__file__).parent.parent / "config" / "llm_config.yaml"
        prompt_dir = config_path.parent
        with open(config_path, "r") as f:
            # Using the same 'product_search' config for the model, but a different prompt
            self.llm_config = yaml.safe_load(f).get("product_search", {})

        prompt_file_path = prompt_dir / "prompts/product_filter.txt"
        with open(prompt_file_path, "r") as f:
            self.prompt_template = f.read()

        model_name = self.llm_config.get("model", "gemini-1.5-flash")
        self.model = genai.GenerativeModel(model_name)
        logger.info(f"FilterService initialized. Model: {model_name}")

    async def filter_products_with_llm(self, products: List[Dict[str, Any]], search_query: str) -> List[str]:
        """
        Uses an LLM to filter a list of products based on a search query.
        Returns a list of 'random_key's for the relevant products.
        """
        if not products:
            return []

        # Prepare the data for the prompt
        product_list_for_prompt = [
            {"name": p.get("name"), "random_key": p.get("random_key")} for p in products
        ]
        product_list_json = json.dumps(product_list_for_prompt, indent=2, ensure_ascii=False)

        prompt = self.prompt_template.format(
            search_query=search_query,
            product_list_json=product_list_json
        )

        try:
            logger.info(f"Sending request to LLM to filter products for query: '{search_query}'")
            response = await self.model.generate_content_async(prompt)

            response_text = response.text.strip()
            logger.debug(f"Raw LLM filter response: {response_text}")

            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            logger.info("Parsing JSON response from LLM filter.")
            filtered_keys = json.loads(response_text)

            if isinstance(filtered_keys, list) and all(isinstance(k, str) for k in filtered_keys):
                logger.info(f"LLM filtered down to {len(filtered_keys)} relevant products.")
                return filtered_keys
            else:
                logger.warning(f"LLM response was not a list of strings: {filtered_keys}")
                return []

        except Exception as e:
            logger.error(f"An error occurred during LLM product filtering: {e}", exc_info=True)
            # In case of an error, return an empty list to avoid breaking the main flow
            return []

filter_service = FilterService()
