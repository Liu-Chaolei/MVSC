from typing import Any

from pydantic import BaseModel


class BaseComponents(BaseModel):
    # pipeline cls
    pipeline_cls: Any = None

    # transformer: the model that should be trained
    transformer: Any = None
