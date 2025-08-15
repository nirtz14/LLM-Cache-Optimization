"""Basic sanity tests for the project skeleton.

These tests verify that the modules in the `project` package import
correctly and that the placeholder classes can be instantiated.  They
provide a foundation on which more thorough unit and integration tests
should be built as the project is implemented.
"""

from project.config import Config
from project.policy import ThresholdPolicy
from project.context import ContextChain
from project.adapter import SemanticCache


def test_config_defaults() -> None:
    """Check that the default configuration values are sensible."""
    cfg = Config()
    assert 0.0 <= cfg.tau <= 1.0
    assert isinstance(cfg.context_enabled, bool)
    assert cfg.cache_size > 0


def test_threshold_policy_accept() -> None:
    """Verify the threshold policy acceptance logic."""
    policy = ThresholdPolicy(tau=0.5)
    assert policy.accept(0.7) is True
    assert policy.accept(0.4) is False


def test_context_chain_same_chain() -> None:
    """Ensure that context chains match when IDs are equal or None."""
    c1 = ContextChain(id="abc")
    c2 = ContextChain(id="abc")
    c3 = ContextChain(id="xyz")
    c_global = ContextChain(id=None)
    assert c1.same_chain(c2) is True
    assert c1.same_chain(c3) is False
    # None matches any context
    assert c1.same_chain(c_global) is True
    assert c_global.same_chain(c3) is True


def test_semantic_cache_put_get() -> None:
    """Test basic put/get behaviour of the semantic cache skeleton."""
    cfg = Config(tau=0.6)
    policy = ThresholdPolicy(tau=cfg.tau)
    cache = SemanticCache(config=cfg, policy=policy)

    query = "What is AI?"
    response = "Artificial intelligence is the simulation of human intelligence."
    context = ContextChain(id="c1")
    similarity = 0.8

    cache.put(query, response, context, similarity)

    # Should hit when similarity is above tau
    hit, resp = cache.get(query, context, similarity=0.75)
    assert hit is True
    assert resp == response

    # Should miss when context is different
    other_context = ContextChain(id="c2")
    hit, resp = cache.get(query, other_context, similarity=0.75)
    assert hit is False
    assert resp is None

    # Should miss when similarity is below threshold
    hit, resp = cache.get(query, context, similarity=0.55)
    assert hit is False
    assert resp is None
