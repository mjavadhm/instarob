from pydantic import BaseModel, Field
from typing import List, Dict

class BoundingBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int

class Frame(BaseModel):
    rank: int
    timestamp_seconds: float
    description: str
    bounding_box_percent: BoundingBox

class FrameAnalysis(BaseModel):
    file_name: str
    product_category: str
    best_frames: List[Frame]
