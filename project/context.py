"""
Context chain utilities for semantic caching.

This module provides functions and classes to manage conversation context
identifiers.  When a cached response is stored it may be associated with
 a context ID (such as a conversation identifier).  Only queries with
matching context IDs should reuse that response.  This prevents
answering a question in one conversation with a response from a
semantically similar but unrelated conversation.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ContextChain:
    """Tracks a conversation context identifier.

    In its simplest form a context identifier is any hashable object that
    uniquely identifies a dialogue or conversation.  In production this
    might be a UUID or a running integer.  Here we treat it as an opaque
    string or integer.
    """

    id: Optional[str] = None

    def same_chain(self, other: "ContextChain") -> bool:
        """Return True if this context is considered the same as another.

        Two context chains are considered the same if both have the same
        non-null identifier.  If either context has no identifier (`None`)
        then it is treated as a global context and matches any other.

        Parameters
        ----------
        other: ContextChain
            The context to compare against.

        Returns
        -------
        bool
            True if the contexts are compatible; False otherwise.
        """
        if self.id is None or other.id is None:
            return True
        return self.id == other.id
