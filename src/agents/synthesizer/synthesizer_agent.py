from src.application.dto.agent_io import CollectorOutput, SynthesizerOutput
from src.domain.models.evidence_chunk import EvidenceChunk


class SynthesizerAgent:
    async def run(self, collected: CollectorOutput, retrieved: list[EvidenceChunk]) -> SynthesizerOutput:
        titles = [doc.title for doc in collected.documents[:3]]
        claims = [f"Key signal from source: {title}" for title in titles]
        comparisons = ["Academic sources emphasize rigor, blogs emphasize implementation speed."]
        questions = ["What production constraints affect adoption timelines?"]
        return SynthesizerOutput(
            claims=claims,
            comparisons=comparisons,
            open_questions=questions,
            evidence=retrieved,
        )
