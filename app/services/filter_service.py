import yaml
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import google.generativeai as genai
from google.generativeai.types import GenerationResponse

from app.core.logging import get_logger

logger = get_logger()

class FilterService:
    def __init__(self):
        # Load LLM and prompt configuration
        config_path = Path(__file__).parent.parent / "config" / "llm_config.yaml"
        prompt_dir = config_path.parent
        with open(config_path, "r") as f:
            self.llm_config = yaml.safe_load(f).get("product_search", {})

        prompt_file_path = prompt_dir / "prompts/product_filter.txt"
        with open(prompt_file_path, "r") as f:
            self.prompt_template = f.read()

        model_name = self.llm_config.get("model", "gemini-1.5-flash")
        self.model = genai.GenerativeModel(model_name)
        logger.info(f"FilterService initialized. Model: {model_name}")

    async def filter_products_with_llm(self, products: List[Dict[str, Any]], search_query: str) -> Tuple[List[str], int, int]:
        """
        Uses an LLM to filter a list of products based on a search query.
        Returns a tuple containing the list of 'random_key's, prompt tokens, and response tokens.
        """
        try:
            prompt_tokens = 0
            response_tokens = 0
            if not products:
                return [], prompt_tokens, response_tokens

            product_list_for_prompt = [
                {"name": p.get("name"), "random_key": p.get("random_key")} for p in products
            ]
            product_list_json = json.dumps(product_list_for_prompt, indent=2, ensure_ascii=False)

            prompt = self.prompt_template.format(
                search_query=search_query,
                product_list_json=product_list_json
            )

            # Calculate prompt tokens
            prompt_token_count_result = await self.model.count_tokens_async(prompt)
            prompt_tokens = prompt_token_count_result.total_tokens
            logger.info(f"Filter prompt token count: {prompt_tokens}")

            logger.info(f"Sending request to LLM to filter products for query: '{search_query}'")
            response = await self.model.generate_content_async(prompt)

            # Check for blocked responses or missing content
            if not response.parts:
                logger.error(f"LLM filter response was blocked or empty. Feedback: {response.prompt_feedback}")
                return [], prompt_tokens, response_tokens

            # Log raw response for debugging
            response_text = response.text.strip()
            logger.info(f"Raw LLM filter response: {response_text}")

            # Calculate response tokens
            response_token_count_result = await self.model.count_tokens_async(response_text)
            response_tokens = response_token_count_result.total_tokens
            logger.info(f"Filter response token count: {response_tokens}")

            # Clean and parse the JSON object
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            logger.info("Parsing JSON object response from LLM filter.")
            parsed_json = json.loads(response_text)

            filtered_keys = parsed_json.get("relevant_keys", [])

            if isinstance(filtered_keys, list) and all(isinstance(k, str) for k in filtered_keys):
                logger.info(f"LLM filtered down to {len(filtered_keys)} relevant products.")
                return filtered_keys, prompt_tokens, response_tokens
            else:
                logger.warning(f"LLM response key 'relevant_keys' was not a list of strings: {filtered_keys}")
                return [], prompt_tokens, response_tokens

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed for LLM filter response: {e}", exc_info=True)
            return [], prompt_tokens, response_tokens
        except Exception as e:
            logger.error(f"An error occurred during LLM product filtering: {e}", exc_info=True)
            return [], prompt_tokens, response_tokens

filter_service = FilterService()
