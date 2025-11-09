from pydantic import BaseModel, Field, HttpUrl
from typing import Optional

class ReelIn(BaseModel):
    text: Optional[str] = Field(default=None, max_length=2200)
    url: HttpUrl
