from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelObject(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "local"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelObject] = Field(default_factory=list)
