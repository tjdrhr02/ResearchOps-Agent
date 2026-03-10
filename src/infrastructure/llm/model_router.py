from dataclasses import dataclass, field


@dataclass
class ModelRouter:
    default_model: str
    task_model_map: dict[str, str] = field(default_factory=dict)

    def route(self, task_type: str, preferred_model: str | None = None) -> str:
        if preferred_model:
            return preferred_model
        if task_type in self.task_model_map:
            return self.task_model_map[task_type]
        return self.default_model
