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
import asyncio
import hashlib
import logging
import struct
from typing import Any, Protocol

import httpx

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


class LocalEmbedder:
    """
    sentence-transformers 기반 로컬 임베딩 서비스.
    API 키 없이 동작하며 실제 의미론적 유사도 검색이 가능하다.
    첫 실행 시 모델을 다운로드하고 이후 캐시에서 사용한다.

    기본 모델: all-MiniLM-L6-v2 (384차원, ~80MB)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        logger.info("local_embedder_init model=%s dims=%s", model_name, self._model.get_embedding_dimension())

    async def embed(self, text: str) -> list[float]:
        loop = asyncio.get_event_loop()
        vector = await loop.run_in_executor(None, self._encode_single, text)
        return vector

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._encode_batch, texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await self.embed(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.embed_batch(texts)

    def _encode_single(self, text: str) -> list[float]:
        return self._model.encode(text, convert_to_numpy=True).tolist()

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, convert_to_numpy=True).tolist()


class GoogleEmbedder:
    """
    Google 임베딩 모델을 사용하는 실제 임베딩 서비스.
    httpx로 Generative Language v1beta REST API를 직접 호출한다.

    모델 우선순위: text-embedding-004 → embedding-001
    초기화 시 사용 가능한 첫 번째 모델을 자동으로 선택한다.
    """

    _BASE = "https://generativelanguage.googleapis.com/v1beta/models"
    _CANDIDATES = ["text-embedding-004", "embedding-001"]

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._model: str | None = None  # _resolve()에서 설정

    async def _resolve(self) -> str:
        """사용 가능한 첫 번째 임베딩 모델을 반환한다."""
        if self._model:
            return self._model
        async with httpx.AsyncClient(timeout=10.0) as client:
            for model_id in self._CANDIDATES:
                url = f"{self._BASE}/{model_id}:embedContent"
                resp = await client.post(
                    url,
                    params={"key": self._api_key},
                    json={
                        "model": f"models/{model_id}",
                        "content": {"parts": [{"text": "test"}]},
                    },
                )
                if resp.status_code == 200:
                    self._model = model_id
                    logger.info("google_embedder_resolved model=%s", model_id)
                    return model_id
        raise RuntimeError(
            f"No accessible Google embedding model found. Tried: {self._CANDIDATES}"
        )

    async def embed(self, text: str) -> list[float]:
        model_id = await self._resolve()
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{self._BASE}/{model_id}:embedContent",
                params={"key": self._api_key},
                json={
                    "model": f"models/{model_id}",
                    "content": {"parts": [{"text": text}]},
                },
            )
            resp.raise_for_status()
        return resp.json()["embedding"]["values"]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return list(await asyncio.gather(*[self.embed(t) for t in texts]))

    async def aembed_query(self, text: str) -> list[float]:
        return await self.embed(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.embed_batch(texts)


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
