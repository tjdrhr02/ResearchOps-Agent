"""
search_papers Tool.

학술 논문 검색 Tool — arXiv REST API 연동.
입력: query(검색어), limit(결과 수)
출력: items(SourceDocument 목록)
"""
import asyncio
import hashlib
import logging
import xml.etree.ElementTree as ET
from typing import Any

import httpx
from pydantic import BaseModel, Field

from src.tools.base.tool_contract import ResearchBaseTool

logger = logging.getLogger(__name__)

_ARXIV_URL = "https://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom"}


class SearchPapersInput(BaseModel):
    query: str = Field(description="Research query to search papers for")
    limit: int = Field(default=3, ge=1, le=10, description="Max number of results")


class SearchPapersTool(ResearchBaseTool):
    name: str = "search_papers"
    description: str = (
        "Search for academic papers relevant to the given research query. "
        "Returns a list of paper titles, URLs, and summaries."
    )
    args_schema: type[BaseModel] = SearchPapersInput
    timeout_seconds: float = 15.0
    retry_count: int = 2

    @property
    def output_schema(self) -> dict[str, Any]:
        return {
            "items": "list[SourceDocument]",
            "fields": ["source_id", "source_type", "title", "url", "content", "metadata"],
        }

    async def _execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = payload["query"]
        limit = max(1, min(int(payload.get("limit", 3)), 10))

        logger.info("search_papers_execute query=%s limit=%s", query, limit)

        params = {
            "search_query": f"all:{query}",
            "max_results": limit,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.get(_ARXIV_URL, params=params)
            resp.raise_for_status()

        # arXiv rate-limit 권장: 연속 요청 사이 1초 대기
        await asyncio.sleep(1)

        root = ET.fromstring(resp.text)
        items = []
        for entry in root.findall("atom:entry", _NS):
            title_el = entry.find("atom:title", _NS)
            summary_el = entry.find("atom:summary", _NS)
            id_el = entry.find("atom:id", _NS)
            published_el = entry.find("atom:published", _NS)

            title = (title_el.text or "").strip().replace("\n", " ")
            abstract = (summary_el.text or "").strip().replace("\n", " ")
            arxiv_url = (id_el.text or "").strip()
            published = (published_el.text or "")[:10] if published_el is not None else ""

            authors = [
                (a.find("atom:name", _NS).text or "").strip()
                for a in entry.findall("atom:author", _NS)
                if a.find("atom:name", _NS) is not None
            ]

            source_id = hashlib.md5(arxiv_url.encode()).hexdigest()
            items.append(
                {
                    "source_id": source_id,
                    "source_type": "paper",
                    "title": title,
                    "url": arxiv_url,
                    "content": abstract,
                    "metadata": {
                        "provider": "arxiv",
                        "query": query,
                        "published": published,
                        "authors": ", ".join(authors[:3]),
                    },
                }
            )

        logger.info("search_papers_done query=%s found=%s", query, len(items))
        return {"items": items}
