"""
EmbeddingService.

LangChain Embeddings 인터페이스를 통해 텍스트를 벡터로 변환한다.

설계:
  - EmbeddingsPort: LangChain BaseEmbeddings 호환 Protocol
  - EmbeddingService: 배치 처리, 오류 격리, 차원 로깅
  - HashEmbedder: 의존성 없이 동작하는 결정적 stub (개발/테스트용)

실제 모델 사용 시:
  from langchain_openai import OpenAIEmbeddings
  service = EmbeddingService(model=OpenAIEmbeddings())
"""
import hashlib
import logging
import struct
from typing import Any, Protocol

logger = logging.getLogger(__name__)

_STUB_DIMS = 8  # HashEmbedder 출력 차원


class EmbeddingsPort(Protocol):
    async def aembed_query(self, text: str) -> list[float]:
        ...

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        ...


class EmbeddingService:
    """
    LangChain Embeddings 모델을 감싸는 서비스.
    배치 embed와 단건 embed를 통일된 인터페이스로 제공한다.
    """

    def __init__(self, model: EmbeddingsPort) -> None:
        self.model = model

    async def embed(self, text: str) -> list[float]:
        vector = await self.model.aembed_query(text)
        logger.debug("embed_query dims=%s", len(vector))
        return vector

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = await self.model.aembed_documents(texts)
        logger.info("embed_batch count=%s dims=%s", len(vectors), len(vectors[0]) if vectors else 0)
        return vectors


class HashEmbedder:
    """
    결정적 해시 기반 stub embedder.
    외부 API 없이 개발/테스트 환경에서 사용한다.
    실제 유사도 측정은 정확하지 않으나 파이프라인 흐름 검증에 충분하다.
    """

    def __init__(self, dims: int = _STUB_DIMS) -> None:
        self.dims = dims

    async def embed(self, text: str) -> list[float]:
        return self._hash_to_vector(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    # LangChain EmbeddingsPort 호환 메서드
    async def aembed_query(self, text: str) -> list[float]:
        return self._hash_to_vector(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    def _hash_to_vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        values: list[float] = []
        for i in range(self.dims):
            raw = struct.unpack_from(">I", digest, offset=(i * 4) % (len(digest) - 3))[0]
            values.append((raw % 10000) / 10000.0)
        return values
