from pydantic.dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class LLMPrompt:
    system_prompt: str | None = None
    human_prompt: str = ''
