from pydantic import BaseModel
from typing import List

class SuggestionsOut(BaseModel):
    suggestions: List[str]
