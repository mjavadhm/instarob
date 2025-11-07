from pydantic import BaseModel, Field, HttpUrl

class ReelIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=2200)
    url: HttpUrl
