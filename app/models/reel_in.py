from pydantic import BaseModel, Field, HttpUrl
from typing import Optional

class ReelIn(BaseModel):
    text: Optional[str] = Field(default=None, max_length=3200)
    url: HttpUrl
