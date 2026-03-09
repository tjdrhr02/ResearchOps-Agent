from dataclasses import dataclass


@dataclass
class LangChainClient:
    model_name: str = "gpt-4o-mini"

    async def invoke(self, prompt: str) -> str:
        # Placeholder for LangChain ChatModel integration.
        return f"LLM response for prompt: {prompt[:120]}"
