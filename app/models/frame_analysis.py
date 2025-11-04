from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class BestFrame(BaseModel):
    rank: int
    timestamp_seconds: float
    description: str

class FrameAnalysis(BaseModel):
    identified_product: str
    best_frames: List[BestFrame]
    product_info: Optional[Dict[str, Any]] = Field(None, description="Product information from Torob search.")
