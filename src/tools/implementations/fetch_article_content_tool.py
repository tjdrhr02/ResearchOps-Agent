"""
fetch_article_content Tool.

URL에서 기사 본문을 가져오는 Tool — httpx + BeautifulSoup4 연동.
입력: url(기사 URL)
출력: content(본문 텍스트), title, word_count
"""
import logging
from typing import Any

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from src.tools.base.tool_contract import ResearchBaseTool

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


class FetchArticleContentInput(BaseModel):
    url: str = Field(description="Full URL of the article to fetch content from")


class FetchArticleContentTool(ResearchBaseTool):
    name: str = "fetch_article_content"
    description: str = (
        "Fetch and extract the full text content from a given article URL. "
        "Returns the article title, body text, and word count."
    )
    args_schema: type[BaseModel] = FetchArticleContentInput
    timeout_seconds: float = 15.0
    retry_count: int = 1

    @property
    def output_schema(self) -> dict[str, Any]:
        return {
            "title": "str",
            "content": "str",
            "word_count": "int",
            "url": "str",
        }

    async def _execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = payload.get("url", "")
        if not url:
            raise ValueError("url must be provided")

        logger.info("fetch_article_execute url=%s", url)

        async with httpx.AsyncClient(
            headers=_HEADERS,
            follow_redirects=True,
            timeout=13.0,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # 제목 추출
        title = ""
        if soup.title:
            title = soup.title.get_text(strip=True)

        # 본문 추출: article > main > body 순서로 시도
        content = ""
        for selector in ("article", "main", "[role='main']"):
            tag = soup.select_one(selector)
            if tag:
                content = tag.get_text(separator=" ", strip=True)
                break

        if not content:
            # 폴백: 모든 <p> 태그 텍스트 결합
            paragraphs = soup.find_all("p")
            content = " ".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

        word_count = len(content.split())
        logger.info("fetch_article_done url=%s word_count=%s", url, word_count)

        return {
            "title": title,
            "content": content[:8000],  # 토큰 제한을 위해 최대 8000자
            "word_count": word_count,
            "url": url,
        }
