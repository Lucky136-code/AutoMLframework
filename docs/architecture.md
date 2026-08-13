# mentorml — Architecture Decision Record

## ADR-001: Protocol over ABC for component interfaces

**Status**: Accepted  
**Date**: Phase 0

**Decision**: Component interfaces are defined using `typing.Protocol` (structural subtyping) rather than Abstract Base Classes.

**Reasoning**: ABCs create rigid inheritance hierarchies. Protocols enforce the *shape* of a component without coupling it to mentorml's class tree. A user's custom preprocessor satisfies `Fittable` simply by having `fit()` and `transform()` methods — no `import mentorml` required in their code.

---

## ADR-002: DecisionRecord as first-class citizen

**Status**: Accepted  
**Date**: Phase 0

**Decision**: Every meaningful action in the pipeline must emit a `DecisionRecord`. Print statements and log messages are not sufficient.

**Reasoning**: Structured data enables: (1) report generation, (2) business narration, (3) JSON audit trails, (4) filtering by severity or component. A log message cannot be post-processed programmatically.

---

## ADR-003: pyproject.toml as single source of truth

**Status**: Accepted  
**Date**: Phase 0

**Decision**: All project metadata, build config, and tool configs (ruff, mypy, pytest, coverage) live in `pyproject.toml`. No `setup.py`, `setup.cfg`, `tox.ini`, `.flake8`, or `mypy.ini`.

**Reasoning**: One file is easier to maintain, review, and understand. `pyproject.toml` is the PEP 517/518/621 standard and is the future direction of Python packaging.

---

## ADR-004: ruff over black + flake8 + isort

**Status**: Accepted  
**Date**: Phase 0

**Decision**: `ruff` handles both linting and formatting.

**Reasoning**: ruff is 10–100× faster, removes a dependency on three separate tools, and is configuration-compatible with existing flake8 rules. The performance difference is noticeable in pre-commit hooks.

---

## ADR-005: Rule-based narrator with optional LLM backend

**Status**: Accepted  
**Date**: Phase 0

**Decision**: The `BusinessNarrator` component will use template-based rule logic by default. An LLM backend (OpenAI / Gemini) is an optional plugin enabled via `pip install mentorml[llm]`.

**Reasoning**: (1) No API key required for the core package. (2) Works fully offline. (3) Reproducible output. (4) LLM narration can be layered on top without redesigning the core.

---

## ADR-006: Self-contained HTML reports

**Status**: Accepted  
**Date**: Phase 0

**Decision**: Reports are single `.html` files with embedded Plotly charts. No server required.

**Reasoning**: A stakeholder can open the file, interact with charts, and share it as an email attachment. Dash/Streamlit dashboards require a running server, which is a deployment dependency that most stakeholders cannot handle.
