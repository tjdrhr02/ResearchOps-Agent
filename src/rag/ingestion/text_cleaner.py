"""
TextCleaner.

파싱된 텍스트의 유니코드/공백/특수문자를 정규화한다.

처리 규칙:
  - 유니코드 정규화 (NFC)
  - 제어 문자 제거 (탭/줄바꿈 제외)
  - 연속 공백 → 단일 공백
  - 연속 빈 줄 → 최대 2줄
  - URL 패턴 보존 (제거하지 않음)
  - 광고/subscribe/cookie 상용구 제거
"""
import re
import unicodedata

_AD_PATTERNS = re.compile(
    r"(subscribe\s+now|sign\s+up\s+for|newsletter|cookie\s+policy"
    r"|advertisement|sponsored\s+content|terms\s+of\s+service"
    r"|privacy\s+policy|all\s+rights\s+reserved)",
    flags=re.IGNORECASE,
)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_LEADING_TRAILING_SPACES_PER_LINE = re.compile(r"^[ \t]+|[ \t]+$", flags=re.MULTILINE)


class TextCleaner:
    def clean(self, text: str) -> str:
        if not text:
            return ""

        text = unicodedata.normalize("NFC", text)
        text = _CONTROL_CHARS.sub("", text)
        text = self._remove_ad_lines(text)
        text = _MULTI_SPACE.sub(" ", text)
        text = _LEADING_TRAILING_SPACES_PER_LINE.sub("", text)
        text = _MULTI_NEWLINE.sub("\n\n", text)
        return text.strip()

    def _remove_ad_lines(self, text: str) -> str:
        lines = text.splitlines()
        return "\n".join(
            line for line in lines
            if not _AD_PATTERNS.search(line)
        )
