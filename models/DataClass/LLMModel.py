from pydantic.dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class LLMModel:
    name: str = ''
    api_base: str = ''
    api_key: str = ''
