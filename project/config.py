"""
Configuration definitions for the semantic cache project.

This module defines a simple `Config` dataclass used to hold runtime
parameters such as the similarity threshold (`tau`), whether context
awareness is enabled and the maximum size of the cache.  These values may
later be loaded from environment variables, configuration files or CLI
arguments.
"""

from dataclasses import dataclass, field


@dataclass
class Config:
    """Holds configuration options for the semantic cache.

    Attributes
    ----------
    tau: float
        The similarity threshold used to decide whether a cached entry is
        considered a match for a new query.  Values should be in the range
        [0.0, 1.0], where larger numbers require higher similarity.
    context_enabled: bool
        If True, cache lookups will only consider hits within the same
        conversation context.  Otherwise, cached responses may be reused
        across contexts.
    cache_size: int
        The maximum number of entries to retain in the in-memory cache.
    """

    tau: float = 0.8
    context_enabled: bool = True
    cache_size: int = 1024

    # Additional fields such as embedding_model and storage_path could be
    # included here in the future.

    def clamp_tau(self) -> None:
        """Clamp the tau parameter into the valid [0.0, 1.0] range.

        This method ensures that the `tau` attribute stays within sensible
        bounds.  It modifies the instance in place.
        """
        if self.tau < 0.0:
            self.tau = 0.0
        elif self.tau > 1.0:
            self.tau = 1.0
