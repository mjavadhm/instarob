from pydantic import BaseModel
from typing import List

class BestFrame(BaseModel):
    rank: int
    timestamp_seconds: float
    description: str

class FrameAnalysis(BaseModel):
    identified_product: str
    best_frames: List[BestFrame]
