from pydantic import BaseModel, Field
from typing import Optional

class CaptionAnalysis(BaseModel):
    search_query_persian: Optional[str] = Field(
        None,
        description="The Persian search query extracted from the caption. It's null if no specific product is found."
    )
    confidence: float = Field(
        ...,
        description="The confidence score (0.0 to 1.0) of the extracted search query."
    )
    is_product_focused: bool = Field(
        ...,
        description="Indicates whether the caption is focused on a specific product."
    )
