from pydantic import BaseModel
from typing import Optional


class StoryRequest(BaseModel):
    promt: Optional[str] = "A story about"
    max_new_tokens: int = 40
    temperature: float = 0.7
    top_p: float = 0.9


class StoryResponse(BaseModel):
    story: str
