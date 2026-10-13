import json
from typing import Any, Optional
from uuid import uuid4
from verl.utils.rollout_trace import rollout_trace_op
from .schemas import OpenAIFunctionToolSchema, ToolResponse


class BaseTool:
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        self.config = config
        self.tool_schema = tool_schema or self.get_openai_tool_schema()
        assert self.tool_schema is not None, "Tool schema is not set!"
        self.name = self.tool_schema.function.name
        print(
            json.dumps(
                self.tool_schema.model_dump(exclude_unset=True, exclude_none=True),
                indent=2,
            )
        )

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            return (str(uuid4()), ToolResponse())
        else:
            return (instance_id, ToolResponse())

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        return (ToolResponse(text="Updated the tool state."), 0.0, {})

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        pass
