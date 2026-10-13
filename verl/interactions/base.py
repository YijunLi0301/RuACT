from typing import Any, Optional
from uuid import uuid4


class BaseInteraction:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.name: str = config.get("name", "interaction_agent")

    async def start_interaction(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> str:
        if instance_id is None:
            return str(uuid4())
        else:
            return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        should_terminate_sequence: bool = False
        response_content: str = "Your current result seems acceptable."
        current_turn_score: float = 0.8
        additional_data: dict[str, Any] = {}
        return (
            should_terminate_sequence,
            response_content,
            current_turn_score,
            additional_data,
        )

    async def calculate_score(self) -> float:
        score = 0.0
        return score

    async def finalize_interaction(self) -> None:
        pass
