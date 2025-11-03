from pydantic import BaseModel, Field, HttpUrl
from typing import Optional

class ReelIn(BaseModel):
    caption: str = Field(..., min_length=1, max_length=2200)
    reel_url: HttpUrl
    request_id: Optional[str] = None
