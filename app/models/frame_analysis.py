from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class BestFrame(BaseModel):
    rank: int
    timestamp_seconds: float
    description: str

class FrameAnalysis(BaseModel):
    identified_product: str
    product_description: Optional[str] = Field(None, description="Detailed visual description from the LLM.")
    search_query_persian: Optional[str] = Field(None, description="The Persian query suggested for text-based search.")
    best_frames: List[BestFrame]
    product_info: Optional[List[Dict[str, Any]]] = Field(None, description="Product information from Torob search.")
    prompt_token_count: Optional[int] = Field(None, description="Token count for the input prompt.")
    response_token_count: Optional[int] = Field(None, description="Token count for the generated output.")
    searched_image_urls: Optional[List[str]] = Field(None, description="URLs of the images searched on Torob.")
