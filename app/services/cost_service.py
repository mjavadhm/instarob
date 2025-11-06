import yaml
from pathlib import Path
from app.core.logging import get_logger

logger = get_logger()

class CostService:
    def __init__(self):
        try:
            config_path = Path(__file__).parent.parent / "config" / "model_pricing.yaml"
            with open(config_path, "r") as f:
                self.pricing_data = yaml.safe_load(f)
            logger.info("CostService initialized with model pricing data.")
        except FileNotFoundError:
            self.pricing_data = {}
            logger.error("model_pricing.yaml not found. CostService will not be able to calculate costs.")
        except Exception as e:
            self.pricing_data = {}
            logger.error(f"Error loading model_pricing.yaml: {e}", exc_info=True)

    def calculate_cost(self, model_name: str, prompt_tokens: int, response_tokens: int) -> float:
        """
        Calculates the cost of a single LLM call.
        """
        if not self.pricing_data or model_name not in self.pricing_data:
            logger.warning(f"Pricing information for model '{model_name}' not found. Cost will be reported as 0.")
            return 0.0

        model_pricing = self.pricing_data[model_name]
        input_cost = (prompt_tokens / 1_000_000) * model_pricing.get("input_cost_per_million_tokens", 0)
        output_cost = (response_tokens / 1_000_000) * model_pricing.get("output_cost_per_million_tokens", 0)

        total_cost = input_cost + output_cost
        logger.info(f"Calculated cost for model '{model_name}': Input={input_cost:.6f}, Output={output_cost:.6f}, Total={total_cost:.6f}")
        return total_cost

cost_service = CostService()
