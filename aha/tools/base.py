from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    ok: bool
    data: str | None = None
    warnings: list[str] = field(default_factory=list)
    redactions: list[str] = field(default_factory=list)
    # Deprecated compatibility fields: keep until the next non-MVP release.
    output: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data is None and self.output is not None:
            self.data = self.output
        elif self.output is None and self.data is not None:
            self.output = self.data

        if self.data is None:
            self.data = ""
        if self.output is None:
            self.output = self.data


class Tool(ABC):
    name: str
    description: str
    input_schema: dict[str, Any]
    side_effect: bool = False

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    @abstractmethod
    async def run(self, args: dict[str, Any]) -> ToolResult:
        raise NotImplementedError
