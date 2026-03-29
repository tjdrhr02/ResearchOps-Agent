"""
HtmlParser.

HTML 원본 콘텐츠에서 본문 텍스트만 추출한다.

처리 규칙:
  - <script>, <style>, <nav>, <footer>, <header>,
    <aside>, <form>, <button> 태그 완전 제거
  - <p>, <h1~h6>, <li>, <blockquote> 단락 구분 보존
  - <code>, <pre> 블록은 코드 마커(```...```)로 변환
  - HTML 엔티티 디코딩 (& → &amp; 등)
  - 입력이 HTML이 아닌 경우 원문 그대로 반환
"""
import logging
import re
from html.parser import HTMLParser as StdHTMLParser

from bs4 import BeautifulSoup, Comment

logger = logging.getLogger(__name__)

_REMOVE_TAGS = {
    "script", "style", "nav", "footer", "header",
    "aside", "form", "button", "noscript", "iframe",
    "svg", "img", "figure", "figcaption", "picture",
    "meta", "link", "head",
}

_BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "br", "tr"}


class HtmlParser:
    def parse(self, raw: str) -> str:
        if not self._looks_like_html(raw):
            return raw.strip()

        try:
            return self._extract_with_bs4(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("html_parser_fallback error=%s", exc)
            return self._extract_with_stdlib(raw)

    # ------------------------------------------------------------------
    # BS4 기반 추출 (기본 경로)
    # ------------------------------------------------------------------

    def _extract_with_bs4(self, raw: str) -> str:
        soup = BeautifulSoup(raw, "html.parser")

        # 주석 제거
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            comment.extract()

        # 불필요 태그 제거
        for tag in soup.find_all(_REMOVE_TAGS):
            tag.decompose()

        # 코드 블록 마킹
        for code_tag in soup.find_all(["pre", "code"]):
            code_text = code_tag.get_text()
            code_tag.replace_with(f"\n```\n{code_text}\n```\n")

        lines: list[str] = []
        for element in soup.find_all(True):
            if element.name in _BLOCK_TAGS:
                text = element.get_text(separator=" ").strip()
                if text:
                    if element.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                        lines.append(f"\n## {text}")
                    else:
                        lines.append(text)

        body_text = "\n".join(lines).strip()

        # 추출이 너무 짧으면 전체 텍스트로 fallback
        if len(body_text) < 100:
            body_text = soup.get_text(separator="\n").strip()

        return body_text

    # ------------------------------------------------------------------
    # 표준 라이브러리 기반 추출 (fallback)
    # ------------------------------------------------------------------

    def _extract_with_stdlib(self, raw: str) -> str:
        class _StripParser(StdHTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self._parts: list[str] = []
                self._skip = False

            def handle_starttag(self, tag: str, attrs: list) -> None:
                if tag in _REMOVE_TAGS:
                    self._skip = True

            def handle_endtag(self, tag: str) -> None:
                if tag in _REMOVE_TAGS:
                    self._skip = False

            def handle_data(self, data: str) -> None:
                if not self._skip:
                    self._parts.append(data)

            def get_text(self) -> str:
                return " ".join(self._parts)

        parser = _StripParser()
        parser.feed(raw)
        return parser.get_text().strip()

    @staticmethod
    def _looks_like_html(text: str) -> bool:
        sample = text[:2000].strip()
        return bool(re.search(r"<[a-zA-Z][^>]*>", sample))
