"""
fetch_article_content Tool.

URL에서 기사 본문을 가져오는 Tool.
입력: url(기사 URL)
출력: content(본문 텍스트), title, word_count
"""
import logging
from typing import Any

from pydantic import BaseModel, Field

from src.tools.base.tool_contract import ResearchBaseTool

logger = logging.getLogger(__name__)


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

        # 실제 환경에서는 httpx/playwright 등으로 교체한다.
        mock_body = (
            f"This article from {url} covers recent developments. "
            "The author presents multiple perspectives, citing empirical studies. "
            "Key findings include improved efficiency, reduced latency, and broader adoption."
        )
        word_count = len(mock_body.split())

        logger.info("fetch_article_done url=%s word_count=%s", url, word_count)
        return {
            "title": f"Article from {url}",
            "content": mock_body,
            "word_count": word_count,
            "url": url,
        }
