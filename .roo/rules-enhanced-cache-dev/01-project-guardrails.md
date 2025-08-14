# Project Guardrails
- **Preserve GPTCache Compatibility**: Always extend or wrap existing GPTCache components rather than forking or modifying its internal code. This ensures upgrades remain straightforward and compliance with upstream licenses.

- **Configuration Driven**: All new features must be optional and controlled via configuration files (YAML/ENV/CLI flags). Provide sensible defaults so the project runs without custom configuration.

- **Modular Design**: Prefer pure functions, small modules, and clear boundaries between `cache/`, `core/`, `utils/`, `benchmark/`, and `tests/`. Avoid cross-module coupling.

- **Tests and Documentation**: Every pull request must include unit or integration tests covering the new behaviour and update the relevant documentation or README sections.

- **Secret Handling**: Never commit secrets or API keys. When an API key is missing, automatically switch to a mock LLM mode so that tests and benchmarks remain runnable offline.
