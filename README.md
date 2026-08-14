# mentorml

> **AI Data Scientist Copilot** — explains the *why* behind every ML decision.

[![CI](https://github.com/your-username/mentorml/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/mentorml/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

## What is mentorml?

Most AutoML tools optimise metrics and produce charts — silently.  
`mentorml` is different: it **reasons and narrates like a senior data scientist**, explaining the *why* behind every transformation, model choice, and business insight it generates.

**Core differentiator**: the explainability and business-insight layer is not a feature — it is the architecture.

---

## Features

| Capability | Status |
|-----------|--------|
| **Decision Layer (core abstractions)** | ✅ Phase 0 |
| Intelligent dataset analysis | ✅ Phase 1 |
| Data quality assessment | ✅ Phase 2 |
| Explainable preprocessing | ✅ Phase 3 |
| Feature engineering | ✅ Phase 4 |
| Model selection & tuning | ✅ Phase 5 |
| SHAP / explainability | ✅ Phase 6 |
| Business insight narration | ✅ Phase 7 |
| Interactive HTML reports | ✅ Phase 8 |
| Deployment-ready model export | ✅ Phase 9 |

---

## Installation

```bash
# Core package
pip install mentorml

# With SHAP support
pip install mentorml[shap]

# With interactive reporting
pip install mentorml[report]

# Everything
pip install mentorml[full]

# For development
pip install -e ".[dev]"
```

---

## Quick Start

```python
import mentorml
from mentorml import MentorConfig
from mentorml.core import DecisionLog, DecisionRecord, Severity

print(mentorml.__version__)  # 0.1.0

# Every session starts with a config and a log
config = MentorConfig(target_column="churn", task_type="classification")
log = DecisionLog()
```

See [`examples/quickstart.py`](examples/quickstart.py) for a full walkthrough.

---

## Architecture

mentorml is built around the **Advisor pattern**: every component does two things — transforms data *and* explains why. A `DecisionLog` accumulates structured `DecisionRecord` objects that power reports, business narration, and audit trails.

```
Data → [Analyzer] → [Strategist] → [Executor] → [Narrator]
                        ↑               ↑              ↑
                  Reasons why      Records what   Translates to
                  each step        happened       business language
```

See [`docs/architecture.md`](docs/architecture.md) for Architecture Decision Records.

---

## Development

```bash
git clone https://github.com/your-username/mentorml
cd mentorml
pip install -e ".[dev]"
pre-commit install

# Run tests
pytest tests/

# Lint + format
ruff check mentorml tests
ruff format mentorml tests

# Type check
mypy mentorml
```

---

## License

MIT © mentorml contributors
