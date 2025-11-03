from pydantic import BaseModel
from typing import List
import uuid

class ReelOut(BaseModel):
    request_id: str
    keys: List[uuid.UUID]
