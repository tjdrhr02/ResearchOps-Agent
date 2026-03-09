class CitationBuilder:
    def build(self, title: str, url: str, source_type: str) -> str:
        return f"[{source_type}] {title} - {url}"
