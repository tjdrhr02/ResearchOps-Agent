"""
Source Type Processors.

paper / blog / news 유형별 추가 처리 규칙을 담당한다.
TextCleaner 이후 단계에서 실행된다.
"""
import re
from abc import ABC, abstractmethod


class BaseSourceProcessor(ABC):
    @abstractmethod
    def process(self, text: str, metadata: dict[str, str]) -> tuple[str, dict[str, str]]:
        """
        텍스트와 메타데이터를 수신해 처리 후 반환한다.
        metadata는 보완/추가 가능하며 원본을 변경하지 않는다.
        """
        raise NotImplementedError


# ------------------------------------------------------------------
# Paper Processor
# ------------------------------------------------------------------

_REFERENCES_SECTION = re.compile(
    r"\n(references|bibliography|works\s+cited|참고\s*문헌)[^\n]*\n.*",
    flags=re.IGNORECASE | re.DOTALL,
)

_ARXIV_HEADER = re.compile(
    r"^(arXiv:\S+|Submitted|Accepted|Published|Preprint)[^\n]*\n",
    flags=re.MULTILINE | re.IGNORECASE,
)

_ABSTRACT_MARKER = re.compile(r"\n?(abstract)[:\s]*", flags=re.IGNORECASE)


class PaperProcessor(BaseSourceProcessor):
    """
    학술 논문 처리기.
    - 참고문헌 섹션 제거
    - arXiv 헤더 제거
    - Abstract 섹션 식별 후 metadata에 기록
    """

    def process(self, text: str, metadata: dict[str, str]) -> tuple[str, dict[str, str]]:
        updated_meta = dict(metadata)

        # Abstract 추출 (metadata에 기록)
        abstract = self._extract_abstract(text)
        if abstract:
            updated_meta["abstract"] = abstract[:500]

        # 참고문헌 이후 제거
        text = _REFERENCES_SECTION.sub("", text)

        # arXiv 헤더 제거
        text = _ARXIV_HEADER.sub("", text)

        return text.strip(), updated_meta

    def _extract_abstract(self, text: str) -> str:
        match = _ABSTRACT_MARKER.search(text)
        if not match:
            return ""
        start = match.end()
        # Abstract 이후 첫 번째 빈 줄까지
        chunk = text[start:start + 1000]
        end = chunk.find("\n\n")
        return chunk[:end].strip() if end != -1 else chunk.strip()


# ------------------------------------------------------------------
# Blog Processor
# ------------------------------------------------------------------

_BOILERPLATE_BLOG = re.compile(
    r"(share\s+this\s+(post|article)|follow\s+us|related\s+posts?"
    r"|comments?\s*\(\d+\)|posted\s+by|tags?:\s*\[)",
    flags=re.IGNORECASE,
)

_CODE_BLOCK = re.compile(r"```.*?```", flags=re.DOTALL)


class BlogProcessor(BaseSourceProcessor):
    """
    기술 블로그 처리기.
    - 공유/팔로우 상용구 제거
    - 코드 블록 보존 (제거하지 않음)
    - 헤더 계층 유지
    """

    def process(self, text: str, metadata: dict[str, str]) -> tuple[str, dict[str, str]]:
        updated_meta = dict(metadata)

        # 코드 블록을 임시 마커로 보호
        code_blocks: list[str] = []

        def _save_code(m: re.Match) -> str:
            code_blocks.append(m.group(0))
            return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

        text = _CODE_BLOCK.sub(_save_code, text)

        # 상용구 줄 제거
        lines = text.splitlines()
        lines = [l for l in lines if not _BOILERPLATE_BLOG.search(l)]
        text = "\n".join(lines)

        # 코드 블록 복원
        for idx, block in enumerate(code_blocks):
            text = text.replace(f"__CODE_BLOCK_{idx}__", block)

        updated_meta["has_code"] = "true" if code_blocks else "false"
        return text.strip(), updated_meta


# ------------------------------------------------------------------
# News Processor
# ------------------------------------------------------------------

_BYLINE = re.compile(
    r"^(by\s+[\w][^\n]{0,60}|reporter:\s*[\w][^\n]{0,60}|author:\s*[\w][^\n]{0,60})\n",
    flags=re.MULTILINE | re.IGNORECASE,
)

_DATE_PATTERN = re.compile(
    r"\b(\w+ \d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})\b"
)

_NEWS_BOILERPLATE = re.compile(
    r"(read\s+more:|more\s+from:|trending\s+now|most\s+read|also\s+read|"
    r"get\s+the\s+latest|breaking\s+news)",
    flags=re.IGNORECASE,
)


class NewsProcessor(BaseSourceProcessor):
    """
    뉴스 처리기.
    - 바이라인(기자 이름) 추출 후 metadata에 기록
    - 날짜 추출 후 metadata에 기록
    - 뉴스 상용구 제거
    - 핵심 단락 우선 보존
    """

    def process(self, text: str, metadata: dict[str, str]) -> tuple[str, dict[str, str]]:
        updated_meta = dict(metadata)

        # 바이라인 추출
        byline_match = _BYLINE.search(text)
        if byline_match:
            updated_meta["byline"] = byline_match.group(0).strip()
            text = _BYLINE.sub("", text)

        # 날짜 추출
        date_match = _DATE_PATTERN.search(text)
        if date_match:
            updated_meta["published_date"] = date_match.group(0)

        # 상용구 줄 제거
        lines = text.splitlines()
        lines = [l for l in lines if not _NEWS_BOILERPLATE.search(l)]
        text = "\n".join(lines)

        return text.strip(), updated_meta


# ------------------------------------------------------------------
# 팩토리
# ------------------------------------------------------------------

_PROCESSORS: dict[str, BaseSourceProcessor] = {
    "paper": PaperProcessor(),
    "tech_blog": BlogProcessor(),
    "blog": BlogProcessor(),
    "news": NewsProcessor(),
}


def get_processor(source_type: str) -> BaseSourceProcessor | None:
    return _PROCESSORS.get(source_type.lower())
