from pydantic import BaseModel, Field, HttpUrl
import uuid

class ReelIn(BaseModel):
    caption: str = Field(..., min_length=1, max_length=2200)
    reel_url: HttpUrl
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
