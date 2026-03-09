class ResearchOpsError(Exception):
    """Base error for ResearchOps service."""


class ToolExecutionError(ResearchOpsError):
    """Raised when a tool execution fails."""


class RetrievalError(ResearchOpsError):
    """Raised when retrieval pipeline fails."""
