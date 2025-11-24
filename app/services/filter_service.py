import yaml
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse

from app.core.logging import get_logger

logger = get_logger()

class FilterService:
    def __init__(self):
        # Load LLM and prompt configuration
        config_path = Path(__file__).parent.parent / "config" / "llm_config.yaml"
        prompt_dir = config_path.parent
        with open(config_path, "r") as f:
            self.llm_config = yaml.safe_load(f).get("product_filter", {})

        prompt_file = self.llm_config.get("prompt_file", "prompts/product_filter.txt")
        prompt_file_path = prompt_dir.parent / "config" / prompt_file

        with open(prompt_file_path, "r") as f:
            self.prompt_template = f.read()

        model_name = self.llm_config.get("model", "gemini-1.5-flash")
        self.model = genai.GenerativeModel(model_name)
        logger.info(f"FilterService initialized. Model: {model_name}")

        # Prepare generation config
        self.generation_config = genai.types.GenerationConfig(
            temperature=self.llm_config.get("temperature", 0.7)
        )

    async def filter_products_with_llm(self, products: List[Dict[str, Any]], search_query: str, identified_product: str, product_description: Optional[str]) -> Tuple[List[str], int, int, str]:
        """
        Uses an LLM to filter a list of products based on a search query and identified product.
        Returns a tuple containing the list of 'random_key's, prompt tokens, response tokens, and the model name used.
        """
        model_name = self.llm_config.get("model", "gemini-1.5-flash")
        try:
            prompt_tokens = 0
            response_tokens = 0
            if not products:
                return [], prompt_tokens, response_tokens, model_name

            product_list_for_prompt = []
            for p in products:
                product_data = {
                    "name": p.get("name"),
                    "random_key": p.get("random_key"),
                    "source": p.get("source", "unknown")
                }
                if "rank" in p:
                    product_data["rank"] = p["rank"]
                product_list_for_prompt.append(product_data)
            product_list_json = json.dumps(product_list_for_prompt, indent=2, ensure_ascii=False)

            prompt = self.prompt_template.format(
                search_query=search_query,
                identified_product=identified_product,
                product_description=product_description or identified_product,
                product_list_json=product_list_json
            )

            # Calculate prompt tokens
            prompt_token_count_result = await self.model.count_tokens_async(prompt)
            prompt_tokens = prompt_token_count_result.total_tokens
            logger.info(f"Filter prompt token count: {prompt_tokens}")

            logger.info(f"Sending request to LLM to filter products for query: '{search_query}'")
            response = await self.model.generate_content_async(prompt, generation_config=self.generation_config)

            # Check for blocked responses or missing content
            if not response.parts:
                logger.error(f"LLM filter response was blocked or empty. Feedback: {response.prompt_feedback}")
                return [], prompt_tokens, response_tokens, model_name

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
                return filtered_keys, prompt_tokens, response_tokens, model_name
            else:
                logger.warning(f"LLM response key 'relevant_keys' was not a list of strings: {filtered_keys}")
                return [], prompt_tokens, response_tokens, model_name

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed for LLM filter response: {e}", exc_info=True)
            return [], prompt_tokens, response_tokens, model_name
        except Exception as e:
            logger.error(f"An error occurred during LLM product filtering: {e}", exc_info=True)
            return [], prompt_tokens, response_tokens, model_name

filter_service = FilterService()
