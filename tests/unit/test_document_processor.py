"""
DocumentProcessor 테스트.

HTML 파싱 / 텍스트 정제 / source_type별 처리 / 파이프라인 조율을 검증한다.
"""
import pytest

from src.domain.models.source_document import SourceDocument
from src.rag.ingestion.document_processor import DocumentProcessor
from src.rag.ingestion.html_parser import HtmlParser
from src.rag.ingestion.source_type_processors import (
    BlogProcessor,
    NewsProcessor,
    PaperProcessor,
    get_processor,
)
from src.rag.ingestion.text_cleaner import TextCleaner


# ──────────────────────────────────────────────
# HtmlParser 테스트
# ──────────────────────────────────────────────

def test_html_parser_removes_script_and_style() -> None:
    html = """
    <html><head><style>body{color:red}</style></head>
    <body>
      <script>alert('hi')</script>
      <p>Main content paragraph.</p>
    </body></html>
    """
    result = HtmlParser().parse(html)
    assert "alert" not in result
    assert "color:red" not in result
    assert "Main content paragraph" in result


def test_html_parser_removes_nav_and_footer() -> None:
    html = """
    <nav>Home | About | Contact</nav>
    <main><p>Article body here.</p></main>
    <footer>Copyright 2024</footer>
    """
    result = HtmlParser().parse(html)
    assert "Article body here" in result
    assert "Home | About" not in result
    assert "Copyright 2024" not in result


def test_html_parser_marks_code_blocks() -> None:
    html = "<p>See example:</p><pre><code>def foo(): pass</code></pre>"
    result = HtmlParser().parse(html)
    assert "```" in result
    assert "def foo" in result


def test_html_parser_returns_plain_text_unchanged() -> None:
    plain = "This is plain text with no HTML tags."
    result = HtmlParser().parse(plain)
    assert result == plain


def test_html_parser_handles_empty_string() -> None:
    assert HtmlParser().parse("") == ""


# ──────────────────────────────────────────────
# TextCleaner 테스트
# ──────────────────────────────────────────────

def test_text_cleaner_normalizes_unicode() -> None:
    text = "caf\u00e9 r\u00e9sum\u00e9"
    result = TextCleaner().clean(text)
    assert "café" in result


def test_text_cleaner_removes_control_chars() -> None:
    text = "Hello\x00World\x1fTest"
    result = TextCleaner().clean(text)
    assert "\x00" not in result
    assert "\x1f" not in result
    assert "HelloWorldTest" in result


def test_text_cleaner_collapses_multiple_spaces() -> None:
    text = "too    many     spaces   here"
    result = TextCleaner().clean(text)
    assert "  " not in result


def test_text_cleaner_collapses_multiple_newlines() -> None:
    text = "line1\n\n\n\n\nline2"
    result = TextCleaner().clean(text)
    assert "\n\n\n" not in result
    assert "line1" in result
    assert "line2" in result


def test_text_cleaner_removes_ad_lines() -> None:
    text = "Useful content.\nSubscribe now for more updates!\nMore useful content."
    result = TextCleaner().clean(text)
    assert "Subscribe now" not in result
    assert "Useful content" in result
    assert "More useful content" in result


# ──────────────────────────────────────────────
# PaperProcessor 테스트
# ──────────────────────────────────────────────

def test_paper_processor_removes_references() -> None:
    text = "Introduction text.\n\nReferences\n[1] Smith et al., 2023\n[2] Doe, 2022"
    processor = PaperProcessor()
    result, _ = processor.process(text, {})
    assert "[1] Smith" not in result
    assert "Introduction text" in result


def test_paper_processor_extracts_abstract_to_metadata() -> None:
    text = "Abstract\nThis paper proposes a new method.\n\nIntroduction\nMore text."
    processor = PaperProcessor()
    _, meta = processor.process(text, {})
    assert "abstract" in meta
    assert "new method" in meta["abstract"]


def test_paper_processor_removes_arxiv_header() -> None:
    text = "arXiv:2401.00001v1\nTitle of Paper\nAbstract\nContent here."
    processor = PaperProcessor()
    result, _ = processor.process(text, {})
    assert "arXiv:2401" not in result


# ──────────────────────────────────────────────
# BlogProcessor 테스트
# ──────────────────────────────────────────────

def test_blog_processor_removes_share_boilerplate() -> None:
    text = "Great engineering content.\nShare this post\nMore content here."
    processor = BlogProcessor()
    result, _ = processor.process(text, {})
    assert "Share this post" not in result
    assert "Great engineering content" in result


def test_blog_processor_preserves_code_blocks() -> None:
    text = "Here is code:\n```\ndef hello(): pass\n```\nEnd of post."
    processor = BlogProcessor()
    result, meta = processor.process(text, {})
    assert "def hello" in result
    assert meta["has_code"] == "true"


def test_blog_processor_marks_no_code() -> None:
    text = "Just text, no code here."
    processor = BlogProcessor()
    _, meta = processor.process(text, {})
    assert meta["has_code"] == "false"


# ──────────────────────────────────────────────
# NewsProcessor 테스트
# ──────────────────────────────────────────────

def test_news_processor_extracts_byline() -> None:
    text = "By Jane Doe\nThe latest news article content.\nMore paragraphs."
    processor = NewsProcessor()
    _, meta = processor.process(text, {})
    assert "byline" in meta
    assert "Jane Doe" in meta["byline"]


def test_news_processor_extracts_date() -> None:
    text = "Published March 9, 2026\nArticle content starts here."
    processor = NewsProcessor()
    _, meta = processor.process(text, {})
    assert "published_date" in meta
    assert "2026" in meta["published_date"]


def test_news_processor_removes_trending_boilerplate() -> None:
    text = "News content.\nRead more: other article\nConclusion."
    processor = NewsProcessor()
    result, _ = processor.process(text, {})
    assert "Read more:" not in result
    assert "News content" in result


# ──────────────────────────────────────────────
# get_processor 팩토리 테스트
# ──────────────────────────────────────────────

def test_get_processor_returns_correct_types() -> None:
    assert isinstance(get_processor("paper"), PaperProcessor)
    assert isinstance(get_processor("tech_blog"), BlogProcessor)
    assert isinstance(get_processor("blog"), BlogProcessor)
    assert isinstance(get_processor("news"), NewsProcessor)
    assert get_processor("unknown") is None


# ──────────────────────────────────────────────
# DocumentProcessor 통합 테스트
# ──────────────────────────────────────────────

def _make_doc(source_type: str, content: str, source_id: str = "doc-1") -> SourceDocument:
    return SourceDocument(
        source_id=source_id,
        source_type=source_type,
        title="Test Document",
        url="https://example.com/doc",
        content=content,
        metadata={"provider": "test"},
    )


def test_processor_paper_full_pipeline() -> None:
    html = """
    <html><body>
      <p>Abstract</p>
      <p>This paper studies RAG evaluation techniques.</p>
      <p>Introduction</p>
      <p>Background information here.</p>
      <p>References</p>
      <p>[1] Smith 2023</p>
    </body></html>
    """
    doc = _make_doc("paper", html)
    result = DocumentProcessor().process(doc)
    assert result is not None
    assert result.source_type == "paper"
    assert "[1] Smith 2023" not in result.clean_content
    assert result.word_count > 0
    assert result.metadata["processed"] == "true"


def test_processor_blog_preserves_code() -> None:
    content = "Intro text.\n```\ndef pipeline(): pass\n```\nConclusion."
    doc = _make_doc("tech_blog", content)
    result = DocumentProcessor().process(doc)
    assert result is not None
    assert "def pipeline" in result.clean_content
    assert result.metadata["has_code"] == "true"


def test_processor_news_extracts_date() -> None:
    html = "<p>By Reporter Name</p><p>March 9, 2026</p><p>Main news content here.</p>"
    doc = _make_doc("news", html)
    result = DocumentProcessor().process(doc)
    assert result is not None
    assert "2026" in result.metadata.get("published_date", "")


def test_processor_handles_empty_content() -> None:
    doc = _make_doc("paper", "")
    result = DocumentProcessor().process(doc)
    assert result is not None
    assert result.word_count == 0


def test_processor_batch_skips_failures_and_continues() -> None:
    docs = [
        _make_doc("paper", "<p>Valid content.</p>", source_id="d1"),
        _make_doc("news", "<p>Another valid document.</p>", source_id="d2"),
    ]
    results = DocumentProcessor().process_many(docs)
    assert len(results) == 2
    assert all(r.metadata["processed"] == "true" for r in results)


def test_processor_word_count_in_metadata() -> None:
    doc = _make_doc("paper", "one two three four five")
    result = DocumentProcessor().process(doc)
    assert result is not None
    assert result.word_count == 5
    assert result.metadata["word_count"] == "5"
