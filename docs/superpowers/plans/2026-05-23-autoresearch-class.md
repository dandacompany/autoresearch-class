# autoresearch-class Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a graduate-level教育 repo that lets students drive `/autoresearch` on a weak CIFAR-10 baseline via codex CLI, with MLflow observability, JOURNAL.md narrative, and a static HTML companion dashboard styled with the Dante Labs Chalkboard design system.

**Architecture:** Five independent units (baseline, mlflow, journal, viz, curriculum) wrap an external `/autoresearch` skill that owns the loop engine. `CURRICULUM.md` instructs codex to walk STAGES/00..07 Socratically. `viz/companion.py` rebuilds a static HTML dashboard after each iteration from `results-log.tsv` + `JOURNAL.md` + `git log`.

**Tech Stack:** Python 3.11+, PyTorch (CPU/MPS), torchvision, MLflow 2.x (docker-compose), pytest, Jinja2 for HTML templating, codex CLI (OAuth), `/autoresearch` skill (already installed).

**Spec:** `docs/superpowers/specs/2026-05-23-autoresearch-class-design.md`

---

## File Structure

```
autoresearch-class/
├── README.md                          # student entry point
├── CURRICULUM.md                      # codex meta directive
├── pyproject.toml
├── .gitignore
├── .python-version
├── STAGES/
│   ├── 00-setup.md
│   ├── 01-skill-tour.md
│   ├── 02-dataset.md
│   ├── 03-mlflow.md
│   ├── 04-baseline.md
│   ├── 05-journal.md
│   ├── 06-loop.md
│   └── 07-debrief.md
├── baseline/
│   ├── __init__.py
│   ├── model.py                       # weak CNN
│   ├── data.py                        # CIFAR-10 loader
│   ├── train.py                       # train + MLflow log
│   └── score.py                       # --print-acc → stdout single float
├── mlflow/
│   ├── docker-compose.yml
│   └── README.md
├── journal/
│   ├── JOURNAL.md
│   └── TEMPLATE.md
├── viz/
│   ├── companion.py
│   ├── _templates/
│   │   └── companion.html.j2
│   └── _static/
│       └── chalkboard.css
├── tests/
│   ├── __init__.py
│   ├── test_model.py
│   ├── test_data.py
│   ├── test_score_print_acc.py
│   ├── test_companion_builder.py
│   ├── test_companion_design_lint.py
│   ├── test_stages_lint.py
│   └── test_fake_iter_e2e.py
├── scripts/
│   └── fake_iter.py
├── .claude/
│   └── settings.json
└── docs/
    ├── INSTRUCTOR.md
    └── superpowers/
        ├── specs/2026-05-23-autoresearch-class-design.md  # already exists
        └── plans/2026-05-23-autoresearch-class.md         # this file
```

---

## Phase 0 — Repo Bootstrap

### Task 0.1: Initialize repo and metadata files

**Files:** `.gitignore`, `.python-version`, `pyproject.toml`, `README.md`

- [ ] **Step 1: Initialize git**

```bash
cd /Users/dante/workspace/dante-code/class/autoresearch
git init && git branch -m main
```

- [ ] **Step 2: Write `.gitignore`**

```
__pycache__/
*.py[cod]
.venv/
.pytest_cache/
.ruff_cache/
data/
*.pt
*.pth
mlflow/mlruns/
mlflow/mlartifacts/
mlflow/postgres-data/
viz/companion.html
.autoresearch/
.vscode/
.idea/
.DS_Store
```

- [ ] **Step 3: Write `.python-version`**

```
3.11
```

- [ ] **Step 4: Write `pyproject.toml`**

```toml
[project]
name = "autoresearch-class"
version = "0.1.0"
description = "Graduate-level hands-on for Karpathy-style autoresearch on CIFAR-10"
requires-python = ">=3.11"
dependencies = [
  "torch>=2.2",
  "torchvision>=0.17",
  "mlflow>=2.10",
  "jinja2>=3.1",
  "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.4"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 5: Write `README.md`**

````markdown
# autoresearch-class

대학원 머신러닝 강의용 핸즈온: 카파시 스타일 autoresearch로 CIFAR-10 모델을 자율 개선합니다.

## Quick start

```bash
git clone <this-repo> autoresearch-class
cd autoresearch-class
codex
```

그 다음 codex에게 한 줄로 지시하세요:

> `@CURRICULUM.md 대로 진행해줘`

codex가 단계별로 안내합니다. 환경 점검 → MLflow 기동 → 베이스라인 학습 → `/autoresearch` 자율 루프 → 회고.

## 필요한 것

- Python 3.11+
- Docker (MLflow용)
- codex CLI (OAuth 인증된 상태)
- `/autoresearch` 스킬 설치
- (선택) NVIDIA GPU 또는 Apple Silicon MPS — 없어도 CPU로 진행 가능
````

- [ ] **Step 6: Commit**

```bash
git add .gitignore .python-version pyproject.toml README.md
git commit -m "chore: bootstrap repo skeleton"
```

---

## Phase 1 — Baseline ML Code (TDD-driven)

### Task 1.1: Weak CNN model

**Files:** `baseline/__init__.py`, `baseline/model.py`, `tests/__init__.py`, `tests/test_model.py`

- [ ] **Step 1: Write the failing test** — `tests/test_model.py`

```python
import torch

from baseline.model import WeakCNN


def test_weak_cnn_forward_shape():
    model = WeakCNN()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10), f"expected (2, 10), got {tuple(y.shape)}"


def test_weak_cnn_has_no_batchnorm():
    """Baseline is intentionally weak — leave BatchNorm as an autoresearch opportunity."""
    model = WeakCNN()
    bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert bn_layers == [], "baseline must not include BatchNorm"
```

- [ ] **Step 2: Run test, verify fail**

```bash
pytest tests/test_model.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create empty packages**

```bash
touch baseline/__init__.py tests/__init__.py
```

- [ ] **Step 4: Implement `baseline/model.py`**

```python
"""Weak CIFAR-10 CNN — intentionally leaves room for autoresearch to improve.

Known gaps (do not pre-fix; that is the agent's job):
- no BatchNorm
- no Dropout
- only 2 conv layers, narrow channels
"""

import torch
import torch.nn as nn


class WeakCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
```

- [ ] **Step 5: Run tests, verify pass**

```bash
pytest tests/test_model.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add baseline/__init__.py baseline/model.py tests/__init__.py tests/test_model.py
git commit -m "feat(baseline): weak CNN with shape test and no-batchnorm guard"
```

---

### Task 1.2: CIFAR-10 data loader

**Files:** `baseline/data.py`, `tests/test_data.py`

- [ ] **Step 1: Write the failing test** — `tests/test_data.py`

```python
from baseline.data import get_loaders


def test_get_loaders_shapes_and_classes(tmp_path):
    train_loader, val_loader = get_loaders(data_root=str(tmp_path), batch_size=4, download=True)
    xb, yb = next(iter(train_loader))
    assert xb.shape == (4, 3, 32, 32)
    assert yb.shape == (4,)
    assert yb.min() >= 0 and yb.max() <= 9


def test_get_loaders_no_augmentation():
    """Baseline must have no data augmentation — autoresearch opportunity."""
    from baseline.data import _train_transform
    from torchvision import transforms

    ops = _train_transform().transforms
    forbidden = (transforms.RandomCrop, transforms.RandomHorizontalFlip, transforms.ColorJitter)
    assert not any(isinstance(op, forbidden) for op in ops)
```

- [ ] **Step 2: Run test, verify fail**

```bash
pytest tests/test_data.py -v
```

- [ ] **Step 3: Implement `baseline/data.py`**

```python
"""CIFAR-10 loader — intentionally no augmentation (autoresearch opportunity)."""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def _train_transform() -> transforms.Compose:
    # NO augmentation on purpose.
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    ])


def _val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    ])


def get_loaders(
    data_root: str = "./data",
    batch_size: int = 16,
    download: bool = True,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    Path(data_root).mkdir(parents=True, exist_ok=True)
    train = datasets.CIFAR10(data_root, train=True, download=download, transform=_train_transform())
    val = datasets.CIFAR10(data_root, train=False, download=download, transform=_val_transform())
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_data.py -v
```

Expected: 2 passed (first run downloads ~170MB).

- [ ] **Step 5: Commit**

```bash
git add baseline/data.py tests/test_data.py
git commit -m "feat(baseline): CIFAR-10 loader with no-augmentation guard"
```

---

### Task 1.3: Training script with MLflow logging

**Files:** `baseline/train.py`

- [ ] **Step 1: Implement `baseline/train.py`**

```python
"""Train the weak CNN on CIFAR-10 and log to MLflow.

Hardcoded hyperparameters on purpose — autoresearch edits the source, not CLI flags.

Known weak choices (leave them; autoresearch will improve):
- SGD without momentum
- lr 0.01 fixed (no scheduler)
- batch_size 16
- epochs 5
"""

import os
import subprocess
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim

from baseline.data import get_loaders
from baseline.model import WeakCNN


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return "unknown"


def main() -> None:
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("autoresearch-class")

    device = _device()
    model = WeakCNN().to(device)
    train_loader, val_loader = get_loaders(batch_size=16, download=True)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    epochs = 5

    with mlflow.start_run() as run:
        mlflow.log_params({
            "model": "WeakCNN",
            "optimizer": "SGD",
            "lr": 0.01,
            "batch_size": 16,
            "epochs": epochs,
        })
        mlflow.set_tag("commit_sha", _git_sha())
        mlflow.set_tag("experiment_iter", os.environ.get("AUTORESEARCH_ITER", "0"))

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= len(train_loader.dataset)

            model.eval()
            correct, total, val_loss = 0, 0, 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    val_loss += criterion(logits, yb).item() * xb.size(0)
                    correct += (logits.argmax(1) == yb).sum().item()
                    total += xb.size(0)
            val_loss /= total
            val_acc = correct / total

            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc},
                step=epoch,
            )
            print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        ckpt_path = Path("data/last_model.pt")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        mlflow.log_artifact(str(ckpt_path))
        mlflow.log_artifact("baseline/model.py")

        Path(".autoresearch").mkdir(exist_ok=True)
        Path(".autoresearch/last_val_acc.txt").write_text(f"{val_acc:.6f}\n")
        print(f"[FINAL] val_acc={val_acc:.4f}  run_id={run.info.run_id}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke check**

```bash
python -c "from baseline.train import _device, _git_sha; print(_device(), _git_sha())"
```

Expected: prints a device string and either a sha or "unknown".

- [ ] **Step 3: Commit**

```bash
git add baseline/train.py
git commit -m "feat(baseline): train.py with MLflow logging and intentional weak defaults"
```

---

### Task 1.4: Scoring script with `--print-acc` (autoresearch Metric command)

**Files:** `baseline/score.py`, `tests/test_score_print_acc.py`

- [ ] **Step 1: Write the failing test** — `tests/test_score_print_acc.py`

```python
import os
import subprocess
from pathlib import Path


def test_print_acc_outputs_single_float(tmp_path):
    """score.py --print-acc must emit ONE float on stdout — autoresearch Metric requirement."""
    acc_file = tmp_path / "last_val_acc.txt"
    acc_file.write_text("0.6234\n")

    repo_root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "AUTORESEARCH_ACC_FILE": str(acc_file)}
    result = subprocess.run(
        ["python", "-m", "baseline.score", "--print-acc"],
        capture_output=True, text=True, cwd=repo_root, env=env,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout.strip()
    assert "\n" not in out, f"expected single line, got: {out!r}"
    value = float(out)
    assert 0.0 <= value <= 1.0
```

- [ ] **Step 2: Run test, verify fail**

```bash
pytest tests/test_score_print_acc.py -v
```

- [ ] **Step 3: Implement `baseline/score.py`**

```python
"""Scoring utility.

`python -m baseline.score --print-acc` — emits the last val_acc as a single float
on stdout. This is the Metric command for autoresearch and MUST output exactly
one parseable number with no extra lines.
"""

import argparse
import os
import sys
from pathlib import Path


def _read_last_acc() -> float:
    path = Path(os.environ.get("AUTORESEARCH_ACC_FILE", ".autoresearch/last_val_acc.txt"))
    if not path.exists():
        print(f"error: {path} not found — run baseline.train first", file=sys.stderr)
        sys.exit(2)
    return float(path.read_text().strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-acc", action="store_true",
                        help="emit last val_acc as a single float on stdout")
    args = parser.parse_args()

    if args.print_acc:
        acc = _read_last_acc()
        print(f"{acc:.6f}")
        return

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_score_print_acc.py -v
```

- [ ] **Step 5: Commit**

```bash
git add baseline/score.py tests/test_score_print_acc.py
git commit -m "feat(baseline): score.py --print-acc as autoresearch Metric command"
```

---

## Phase 2 — MLflow Infrastructure

### Task 2.1: docker-compose for MLflow

**Files:** `mlflow/docker-compose.yml`, `mlflow/README.md`

- [ ] **Step 1: Write `mlflow/docker-compose.yml`**

```yaml
services:
  mlflow-db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow
      POSTGRES_DB: mlflow
    volumes:
      - ./postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mlflow"]
      interval: 5s
      timeout: 3s
      retries: 10

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.14.1
    depends_on:
      mlflow-db:
        condition: service_healthy
    command: >
      mlflow server
      --backend-store-uri postgresql://mlflow:mlflow@mlflow-db:5432/mlflow
      --default-artifact-root /mlartifacts
      --host 0.0.0.0
      --port 5000
    ports:
      - "5000:5000"
    volumes:
      - ./mlartifacts:/mlartifacts
```

- [ ] **Step 2: Write `mlflow/README.md`**

````markdown
# MLflow server (self-hosted)

```bash
cd mlflow
docker compose up -d
open http://localhost:5000
docker compose down            # stop (preserves data)
docker compose down -v         # reset (drops postgres volume)
```

학습 스크립트는 `MLFLOW_TRACKING_URI=http://127.0.0.1:5000`을 기본값으로 사용합니다.
````

- [ ] **Step 3: Smoke test (manual, optional)**

```bash
cd mlflow && docker compose up -d
sleep 8
curl -fsS http://127.0.0.1:5000/health
docker compose down
```

- [ ] **Step 4: Commit**

```bash
git add mlflow/docker-compose.yml mlflow/README.md
git commit -m "feat(mlflow): self-hosted server via docker-compose"
```

---

## Phase 3 — Journal + Visual Companion

### Task 3.1: Journal template and empty journal

**Files:** `journal/TEMPLATE.md`, `journal/JOURNAL.md`

- [ ] **Step 1: Write `journal/TEMPLATE.md`**

```markdown
## iter {N} — {KEEP|DISCARD|CRASH}

- **commit**: `{short_sha}`
- **mlflow_run**: `{run_id}` ([open](http://localhost:5000/#/experiments/0/runs/{run_id}))
- **hypothesis**: {한 문장 가설}
- **change**: {한 줄 요약}
- **metric**: {baseline_acc → new_acc, Δ}
- **verdict**: {해석}
- **next**: {다음 시도}
```

- [ ] **Step 2: Write `journal/JOURNAL.md`**

```markdown
# autoresearch-class journal

이 파일은 `/autoresearch` 루프 중 각 iteration 종료 후 codex가 append합니다.
형식은 `journal/TEMPLATE.md`를 따릅니다. 가장 최근 iter가 맨 아래.

---
```

- [ ] **Step 3: Commit**

```bash
git add journal/TEMPLATE.md journal/JOURNAL.md
git commit -m "feat(journal): entry template and empty journal"
```

---

### Task 3.2: Chalkboard CSS asset

**Files:** `viz/_static/chalkboard.css`

- [ ] **Step 1: Write `viz/_static/chalkboard.css`**

```css
/* Dante Labs Chalkboard — adapted for web companion */

:root {
  --bg-chalkboard: #2f4f4f;
  --bg-chalkboard-deep: #243f3f;
  --bg-chalkboard-raised: #3d5c5c;

  --ink-primary: #ffffff;
  --ink-muted: #d8d5c9;
  --ink-subtle: #9bafaf;

  --accent-tomato: #ff6347;
  --accent-yellow: #ffd700;

  --stroke-chalk: rgba(255, 255, 255, 0.72);
  --stroke-chalk-soft: rgba(255, 255, 255, 0.36);
  --dust-chalk: rgba(255, 255, 255, 0.06);

  --semantic-success: #7fb069;
  --semantic-danger: #ff6347;
  --semantic-warn: #ffd700;
  --semantic-info: #9bc4e2;

  --safe-inset-x: 96px;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background: var(--bg-chalkboard);
  color: var(--ink-muted);
  font-family: "Gowun Dodum", sans-serif;
  line-height: 1.7;
}

.page {
  max-width: 1200px;
  margin: 0 auto;
  padding: 48px var(--safe-inset-x);
}

h1.title-display {
  font-family: "Nanum Pen Script", "Caveat", cursive;
  color: var(--ink-primary);
  font-size: clamp(48px, 6vw, 88px);
  line-height: 1.1;
  transform: rotate(-1.2deg);
  margin: 0 0 24px 0;
}

.meta {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ink-subtle);
}

.stat-hero {
  font-family: "Caveat", "Nanum Pen Script", cursive;
  font-weight: 700;
  font-size: clamp(96px, 12vw, 192px);
  line-height: 0.95;
  color: var(--accent-yellow);
}
.stat-hero .unit {
  font-family: "Gowun Dodum", sans-serif;
  font-weight: 700;
  font-size: 20px;
  letter-spacing: 0.04em;
  margin-left: 8px;
  color: var(--ink-muted);
}

.iter-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 24px;
  margin-top: 32px;
}
@media (max-width: 960px) {
  .iter-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}
@media (max-width: 640px) {
  .iter-grid {
    grid-template-columns: 1fr;
  }
}

.chalk-card {
  background: var(--bg-chalkboard-raised);
  border: 1.5px solid var(--stroke-chalk);
  border-radius: 4px;
  padding: 20px;
  box-shadow: 0 0 24px var(--dust-chalk);
}
.chalk-card h3 {
  font-family: "Nanum Pen Script", cursive;
  font-weight: 400;
  color: var(--ink-primary);
  margin: 0 0 8px 0;
  font-size: 28px;
}

.badge {
  display: inline-block;
  padding: 2px 10px;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  border: 1.5px solid var(--stroke-chalk);
  border-radius: 999px;
}
.badge.keep {
  background: var(--accent-tomato);
  color: var(--ink-primary);
  border-color: var(--accent-tomato);
}
.badge.discard {
  background: transparent;
  color: var(--ink-muted);
}
.badge.crash {
  background: transparent;
  color: var(--ink-muted);
  border-style: dashed;
}

.callout-info,
.callout-warn {
  border: 2px solid var(--stroke-chalk);
  border-radius: 4px;
  padding: 12px 16px;
  margin: 12px 0;
  font-family: "JetBrains Mono", monospace;
  font-size: 13px;
  background: var(--bg-chalkboard-deep);
  color: var(--ink-muted);
}
.callout-warn {
  border-color: var(--accent-yellow);
}

.divider {
  height: 1px;
  background: var(--stroke-chalk-soft);
  border: none;
  margin: 32px 0;
}

.highlight-yellow {
  color: var(--accent-yellow);
}

a {
  color: var(--ink-primary);
  text-decoration: underline;
  text-decoration-color: var(--accent-yellow);
}
```

- [ ] **Step 2: Commit**

```bash
git add viz/_static/chalkboard.css
git commit -m "feat(viz): chalkboard.css design tokens"
```

---

### Task 3.3: Companion Jinja2 template

**Files:** `viz/_templates/companion.html.j2`

- [ ] **Step 1: Write the template**

```jinja
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>autoresearch progress</title>
  <link href="https://fonts.googleapis.com/css2?family=Nanum+Pen+Script&family=Gowun+Dodum:wght@400;700&family=Caveat:wght@700&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="_static/chalkboard.css">
</head>
<body>
  <main class="page">
    <header>
      <h1 class="title-display">autoresearch progress</h1>
      <div class="meta">iter {{ iter_current }} / {{ iter_total }} · best {{ "%.1f"|format(best_acc * 100) }}%</div>
    </header>

    <section>
      <div class="stat-hero">{{ "%.1f"|format(best_acc * 100) }}<span class="unit">%</span></div>
      <div class="meta">best validation accuracy so far</div>
    </section>

    <hr class="divider">

    <section class="iter-grid">
      {% for it in iters %}
      <article class="chalk-card" style="transform: rotate({{ it.rotation }}deg);">
        <h3>iter {{ it.n }} <span class="badge {{ it.verdict|lower }}">{{ it.verdict }}</span></h3>
        <p><strong>가설.</strong> {{ it.hypothesis }}</p>
        <p><strong>변경.</strong> {{ it.change }}</p>
        <p><strong>결과.</strong> {{ "%.2f"|format(it.acc_before * 100) }}% → <span class="highlight-yellow">{{ "%.2f"|format(it.acc_after * 100) }}%</span> (Δ {{ "%+.2f"|format((it.acc_after - it.acc_before) * 100) }}p)</p>
        {% if it.verdict_text %}<p><strong>해석.</strong> {{ it.verdict_text }}</p>{% endif %}
        {% if it.git_sha %}<div class="callout-info">commit {{ it.git_sha }}</div>{% endif %}
        {% if it.mlflow_run %}<div class="meta"><a href="http://localhost:5000/#/experiments/0/runs/{{ it.mlflow_run }}">mlflow run</a></div>{% endif %}
      </article>
      {% endfor %}
    </section>
  </main>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add viz/_templates/companion.html.j2
git commit -m "feat(viz): companion HTML template with chalkboard layout"
```

---

### Task 3.4: Companion builder script (TDD)

**Files:** `viz/__init__.py`, `viz/companion.py`, `tests/test_companion_builder.py`

- [ ] **Step 1: Write the failing test** — `tests/test_companion_builder.py`

```python
from pathlib import Path

from viz.companion import IterEntry, build_html, parse_journal, parse_results_log  # noqa: F401


SAMPLE_JOURNAL = """
# autoresearch-class journal

---

## iter 1 — KEEP

- **commit**: `abc1234`
- **mlflow_run**: `run-xyz-001`
- **hypothesis**: BatchNorm을 conv 뒤에 넣으면 수렴이 안정될 것
- **change**: + nn.BatchNorm2d(16), nn.BatchNorm2d(32)
- **metric**: 0.5820 → 0.6510 (+6.90)
- **verdict**: 예상대로 큰 폭 개선
- **next**: scheduler 도입
"""


SAMPLE_TSV = """iter\tverdict\tacc_before\tacc_after
1\tKEEP\t0.5820\t0.6510
"""


def test_parse_journal_extracts_iter(tmp_path):
    p = tmp_path / "JOURNAL.md"
    p.write_text(SAMPLE_JOURNAL)
    entries = parse_journal(p)
    assert len(entries) == 1
    e = entries[0]
    assert e.n == 1
    assert e.verdict == "KEEP"
    assert e.git_sha == "abc1234"
    assert e.mlflow_run == "run-xyz-001"
    assert "BatchNorm" in e.hypothesis


def test_parse_results_log_extracts_metrics(tmp_path):
    p = tmp_path / "results-log.tsv"
    p.write_text(SAMPLE_TSV)
    rows = parse_results_log(p)
    assert rows[1]["acc_after"] == 0.6510


def test_build_html_writes_file_with_required_markers(tmp_path):
    journal = tmp_path / "JOURNAL.md"
    journal.write_text(SAMPLE_JOURNAL)
    tsv = tmp_path / "results-log.tsv"
    tsv.write_text(SAMPLE_TSV)
    out = tmp_path / "companion.html"

    build_html(journal_path=journal, results_log_path=tsv, output_path=out, iter_total=10)

    html = out.read_text()
    assert "chalkboard.css" in html
    assert "Nanum+Pen+Script" in html
    assert "iter 1" in html
    assert "BatchNorm" in html
    assert "65.10%" in html
```

- [ ] **Step 2: Run test, verify fail**

```bash
pytest tests/test_companion_builder.py -v
```

- [ ] **Step 3: Create `viz/__init__.py`** (empty)

```bash
touch viz/__init__.py
```

- [ ] **Step 4: Implement `viz/companion.py`**

```python
"""Build viz/companion.html from JOURNAL.md + results-log.tsv.

Static builder. No server. Re-run after each autoresearch iteration:

    python -m viz.companion
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from jinja2 import Environment, FileSystemLoader, select_autoescape


@dataclass
class IterEntry:
    n: int
    verdict: str  # KEEP | DISCARD | CRASH
    hypothesis: str = ""
    change: str = ""
    verdict_text: str = ""
    git_sha: str | None = None
    mlflow_run: str | None = None
    acc_before: float = 0.0
    acc_after: float = 0.0
    rotation: float = 0.0


_ITER_HEADER = re.compile(r"^## iter (\d+) — (KEEP|DISCARD|CRASH)\s*$", re.MULTILINE)
_FIELD = re.compile(r"^- \*\*(\w+)\*\*: (.+)$", re.MULTILINE)


def parse_journal(path: Path) -> list[IterEntry]:
    if not path.exists():
        return []
    text = path.read_text()
    entries: list[IterEntry] = []
    headers = list(_ITER_HEADER.finditer(text))
    for idx, m in enumerate(headers):
        start = m.end()
        end = headers[idx + 1].start() if idx + 1 < len(headers) else len(text)
        block = text[start:end]
        fields = {fm.group(1): fm.group(2).strip() for fm in _FIELD.finditer(block)}
        entry = IterEntry(
            n=int(m.group(1)),
            verdict=m.group(2),
            hypothesis=fields.get("hypothesis", ""),
            change=fields.get("change", ""),
            verdict_text=fields.get("verdict", ""),
            git_sha=_strip_backticks(fields.get("commit", "")),
            mlflow_run=_strip_backticks(fields.get("mlflow_run", "")),
        )
        entries.append(entry)
    return entries


def _strip_backticks(s: str) -> str | None:
    s = s.strip()
    if not s:
        return None
    return s.strip("`").strip()


def parse_results_log(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    rows: dict[int, dict] = {}
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            n = int(row["iter"])
            rows[n] = {
                "verdict": row.get("verdict", ""),
                "acc_before": float(row.get("acc_before") or 0.0),
                "acc_after": float(row.get("acc_after") or 0.0),
            }
    return rows


def _assign_rotations(entries: Iterable[IterEntry]) -> None:
    angles = [-1.5, 0.8, -0.4, 1.2, -0.9, 0.6, -1.1, 0.3]
    for i, e in enumerate(entries):
        e.rotation = angles[i % len(angles)]


def build_html(
    journal_path: Path,
    results_log_path: Path,
    output_path: Path,
    iter_total: int = 10,
) -> None:
    entries = parse_journal(journal_path)
    rows = parse_results_log(results_log_path)
    for e in entries:
        if e.n in rows:
            e.acc_before = rows[e.n]["acc_before"]
            e.acc_after = rows[e.n]["acc_after"]
    _assign_rotations(entries)

    iter_current = max((e.n for e in entries), default=0)
    best_acc = max((e.acc_after for e in entries), default=0.0)

    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "_templates"),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("companion.html.j2")
    html = template.render(
        iters=entries,
        iter_current=iter_current,
        iter_total=iter_total,
        best_acc=best_acc,
    )
    output_path.write_text(html)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    build_html(
        journal_path=root / "journal/JOURNAL.md",
        results_log_path=root / ".autoresearch/results-log.tsv",
        output_path=root / "viz/companion.html",
    )
    assert (root / "viz/_static/chalkboard.css").exists(), "missing chalkboard.css"
    print(f"wrote {root / 'viz/companion.html'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests, verify pass**

```bash
pytest tests/test_companion_builder.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add viz/__init__.py viz/companion.py tests/test_companion_builder.py
git commit -m "feat(viz): companion.py static HTML builder"
```

---

### Task 3.5: Design-lint test (Chalkboard rules)

**Files:** `tests/test_companion_design_lint.py`

- [ ] **Step 1: Write the design-lint test**

```python
"""Mechanical guards for Chalkboard rules (spec §7.4)."""

from pathlib import Path
import re


CSS = (Path(__file__).resolve().parents[1] / "viz/_static/chalkboard.css").read_text()
TEMPLATE = (Path(__file__).resolve().parents[1] / "viz/_templates/companion.html.j2").read_text()


def test_no_gradient_in_css():
    assert "linear-gradient" not in CSS
    assert "radial-gradient" not in CSS


def test_no_backdrop_filter():
    assert "backdrop-filter" not in CSS


def test_no_corner_radius_above_8px():
    for m in re.finditer(r"border-radius:\s*([0-9]+)px", CSS):
        assert int(m.group(1)) <= 8, f"border-radius {m.group(0)} exceeds 8px"


def test_chalkboard_bg_color_set():
    assert "#2F4F4F" in CSS


def test_accent_colors_are_tomato_and_yellow():
    assert "#FF6347" in CSS
    assert "#FFD700" in CSS


def test_template_references_chalkboard_css():
    assert "chalkboard.css" in TEMPLATE


def test_template_loads_required_fonts():
    for fam in ("Nanum+Pen+Script", "Gowun+Dodum", "JetBrains+Mono"):
        assert fam in TEMPLATE, f"missing font: {fam}"
```

- [ ] **Step 2: Run tests, verify pass**

```bash
pytest tests/test_companion_design_lint.py -v
```

Expected: 7 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_companion_design_lint.py
git commit -m "test(viz): mechanical Chalkboard design lint"
```

---

## Phase 4 — Curriculum (STAGES + CURRICULUM)

### Task 4.1: STAGES lint test (structure guard)

**Files:** `tests/test_stages_lint.py`

- [ ] **Step 1: Write the failing test**

```python
"""Each STAGES/*.md must have the 4 canonical sections."""

from pathlib import Path

import pytest


STAGES_DIR = Path(__file__).resolve().parents[1] / "STAGES"

REQUIRED_SECTIONS = ["## 목표", "## 사전 점검", "## 소크라테스 질문", "## 통과 조건"]

EXPECTED_FILES = [
    "00-setup.md",
    "01-skill-tour.md",
    "02-dataset.md",
    "03-mlflow.md",
    "04-baseline.md",
    "05-journal.md",
    "06-loop.md",
    "07-debrief.md",
]


@pytest.mark.parametrize("name", EXPECTED_FILES)
def test_stage_file_exists(name):
    assert (STAGES_DIR / name).exists(), f"missing STAGES/{name}"


@pytest.mark.parametrize("name", EXPECTED_FILES)
def test_stage_has_required_sections(name):
    text = (STAGES_DIR / name).read_text()
    for sec in REQUIRED_SECTIONS:
        assert sec in text, f"STAGES/{name} missing section {sec!r}"
```

- [ ] **Step 2: Run, verify all fail**

```bash
pytest tests/test_stages_lint.py -v
```

Expected: 16 failed (8 missing files × 2 tests).

- [ ] **Step 3: Commit**

```bash
git add tests/test_stages_lint.py
git commit -m "test(stages): structural lint for STAGES/*.md"
```

---

### Task 4.2: STAGES/00-setup.md

**Files:** `STAGES/00-setup.md`

- [ ] **Step 1: Write the file**

````markdown
## 목표

수강생의 환경(codex CLI OAuth, `/autoresearch` 스킬, Python 3.11+, Docker)이 갖춰졌는지 확인하고, 작업 디렉토리를 합의한다.

## 사전 점검

```bash
codex --version
which docker && docker version --format '{{.Server.Version}}'
python3 --version
ls -d ~/.claude/plugins/cache/autoresearch 2>/dev/null && echo "autoresearch skill present" || echo "autoresearch skill MISSING"
```

위 명령을 codex가 실행하고 결과를 보고합니다.

## 소크라테스 질문

1. (자동 점검 결과를 보여준 뒤) "진행 모드를 어떻게 할까요?"
   - ① 모든 단계를 codex가 자동 보고 → 수강생은 OK만
   - ② 단계마다 수강생이 직접 명령 입력 → codex는 안내만
2. "작업 디렉토리는 어디로 할까요?" (기본 `~/workspace/autoresearch-class/`)
3. "Docker가 띄워져 있나요? 없으면 Docker Desktop을 켜고 다시 알려주세요."

## 통과 조건

- [ ] codex CLI 인증됨
- [ ] `/autoresearch` 스킬 설치 확인됨
- [ ] Python 3.11+ 확인됨
- [ ] Docker 데몬 동작 중
- [ ] 작업 디렉토리 합의됨
````

- [ ] **Step 2: Commit**

```bash
git add STAGES/00-setup.md
git commit -m "feat(stages): 00-setup — env check"
```

---

### Task 4.3: STAGES/01-skill-tour.md

**Files:** `STAGES/01-skill-tour.md`

- [ ] **Step 1: Write the file**

````markdown
## 목표

`/autoresearch` 스킬의 동작 원리(Modify → Verify → Keep/Discard → Repeat)와 5요소(Goal / Scope / Metric / Direction / Verify) + Guard를 수강생이 자신의 언어로 설명할 수 있게 한다.

## 사전 점검

```bash
cat ~/.claude/plugins/cache/autoresearch/autoresearch/*/skills/autoresearch/SKILL.md | head -80
```

codex가 SKILL.md의 핵심 절(MANDATORY Interactive Setup, The Loop, Critical Rules)을 요약해 1분 안에 보여줍니다.

## 소크라테스 질문

1. "Metric은 반드시 단일 float을 stdout으로 출력해야 합니다. 왜 그래야 할까요? — JSON이나 다중 라인이면 어떤 문제가 생길까요?"
2. "스킬은 한 iter에 한 가지 변경만 적용합니다. 만약 'BatchNorm 추가 + LR 변경'을 한 번에 한다면, 다음 iter에서 무엇을 잃게 될까요?"
3. "Guard와 Metric의 차이를 한 문장으로 정의해 보세요. — '항상 통과해야 하는 것' vs '더 좋아져야 하는 것'."
4. "git 커밋이 곧 메모리라는 말은 무슨 뜻일까요? 만약 커밋 없이 변경했다면 rollback이 어떻게 깨질까요?"

각 질문에 수강생이 답하면 codex는 정답을 즉답하지 않고, 답의 어느 부분이 더 정확해질 수 있는지 1~2 turn 안에 안내합니다.

## 통과 조건

- [ ] Metric 단일 float 제약을 수강생이 설명함
- [ ] one-change-per-iter 원칙을 수강생이 설명함
- [ ] Guard와 Metric의 차이를 한 문장으로 표현함
- [ ] git 커밋 = 메모리의 의미를 수강생이 설명함
````

- [ ] **Step 2: Commit**

```bash
git add STAGES/01-skill-tour.md
git commit -m "feat(stages): 01-skill-tour — Socratic intro to /autoresearch"
```

---

### Task 4.4: STAGES/02-dataset.md

**Files:** `STAGES/02-dataset.md`

- [ ] **Step 1: Write the file**

````markdown
## 목표

CIFAR-10을 다운로드하고, 학습에 들어가기 전 데이터의 정상성을 sanity check 한다.

## 사전 점검

```bash
python -c "from baseline.data import get_loaders; tl, vl = get_loaders(batch_size=4); xb, yb = next(iter(tl)); print('shape:', xb.shape, 'labels:', yb.tolist())"
```

처음 호출 시 ~170MB 다운로드 후 캐시. 두 번째부터는 즉시 로드.

## 소크라테스 질문

1. "다운로드된 데이터는 `./data/`에 저장됩니다. git에 함께 올리면 어떤 문제가 생길까요? 처리 방법은?
   - ① `.gitignore` 에 추가 ② Git LFS ③ 외부 캐시 디렉토리"
2. "학습에 들어가기 전 sanity check로 무엇을 확인해야 할까요?"
   - (정답 컨벤션) 클래스 분포 / 이미지 shape / 정규화 통계
3. "test_data.py 는 'augmentation이 없는지'를 명시적으로 검사합니다. 왜 그런 가드를 둘까요? — augmentation을 베이스라인이 가지고 있으면 autoresearch에 어떤 영향이 있을까요?"

## 통과 조건

- [ ] `./data/cifar-10-batches-py/` 가 존재함
- [ ] `pytest tests/test_data.py -v` 통과
- [ ] 수강생이 sanity check 3가지를 본인의 언어로 말함
````

- [ ] **Step 2: Commit**

```bash
git add STAGES/02-dataset.md
git commit -m "feat(stages): 02-dataset — CIFAR-10 sanity check"
```

---

### Task 4.5: STAGES/03-mlflow.md

**Files:** `STAGES/03-mlflow.md`

- [ ] **Step 1: Write the file**

````markdown
## 목표

self-hosted MLflow 서버를 docker-compose로 띄우고 UI 접속을 확인한다.

## 사전 점검

```bash
cd mlflow && docker compose up -d
sleep 8
curl -fsS http://127.0.0.1:5000/health
```

`OK` 가 출력되면 정상.

## 소크라테스 질문

1. "MLflow를 (a) `mlflow ui` 로컬 단일 프로세스 vs (b) docker-compose self-host 중 어느 쪽으로 띄울까요? 강의에서는 (b)를 권장하는데 그 이유는?"
   - 힌트: 종료 후 재기동 시 run 기록이 보존되는가
2. "MLflow의 'experiment'와 'run'의 차이는 무엇일까요? autoresearch 1회 호출은 1 experiment에 매핑되는 것이 자연스러울까요, 아니면 1 run인가요?"
   - 정답 컨벤션: 1 experiment(autoresearch-class), N runs (iter별 1 run)
3. "수강생이 매 iter 후 MLflow UI에서 무엇을 보면 좋을까요?" (Compare 기능, val_acc 라인 차트, params diff)

## 통과 조건

- [ ] `http://localhost:5000` UI 로드됨
- [ ] `autoresearch-class` experiment가 자동 생성될 준비됨
- [ ] 수강생이 experiment vs run 차이를 설명함
````

- [ ] **Step 2: Commit**

```bash
git add STAGES/03-mlflow.md
git commit -m "feat(stages): 03-mlflow — server bring-up"
```

---

### Task 4.6: STAGES/04-baseline.md

**Files:** `STAGES/04-baseline.md`

- [ ] **Step 1: Write the file**

````markdown
## 목표

의도적으로 약한 베이스라인을 1회 학습하고, val_acc를 0번 iter 기준점으로 기록한다.

## 사전 점검

MLflow가 동작 중이어야 한다.

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 python -m baseline.train
```

학습 시간: MPS 약 5분, CPU 약 10분, NVIDIA GPU 약 1분.

## 소크라테스 질문

1. "베이스라인이 '가지고 있지 말아야 할' 항목을 고르세요. (다중 선택)"
   - ① BatchNorm ② Data augmentation ③ LR scheduler ④ Dropout ⑤ Adam optimizer
   - (정답: 1~5 모두. 이들이 모두 autoresearch가 발견할 개선 기회)
2. "방금 학습 후 출력된 최종 val_acc는 몇 %인가요? (예상: 55~62%)"
3. "MLflow UI에서 방금 run의 params·metrics 탭을 열어보세요. 어떤 값이 보이나요?"
4. "이 정확도가 만족스러우신가요? 75%로 끌어올리려면 무엇을 바꿔야 한다고 생각하시나요? — 답을 메모해두세요. 이후 autoresearch가 같은 아이디어를 시도하는지 비교할 겁니다."

## 통과 조건

- [ ] `./data/last_model.pt` 존재
- [ ] `.autoresearch/last_val_acc.txt` 존재
- [ ] MLflow UI에서 run 1개 확인됨
- [ ] 수강생이 자기 가설을 메모함
````

- [ ] **Step 2: Commit**

```bash
git add STAGES/04-baseline.md
git commit -m "feat(stages): 04-baseline — first training, 0-th iter"
```

---

### Task 4.7: STAGES/05-journal.md

**Files:** `STAGES/05-journal.md`

- [ ] **Step 1: Write the file**

````markdown
## 목표

JOURNAL.md 컨벤션을 이해하고, 빈 companion.html을 한 번 빌드해 둔다.

## 사전 점검

```bash
cat journal/TEMPLATE.md
python -m viz.companion
open viz/companion.html
```

빈 페이지지만 헤더와 디자인 시스템이 적용되어 있어야 한다.

## 소크라테스 질문

1. "autoresearch는 자체적으로 `.autoresearch/results-log.tsv`를 남깁니다. 그런데 사람이 읽는 `JOURNAL.md`를 별도로 두는 이유는 무엇일까요?"
   - 힌트: tsv에는 들어갈 수 없고 journal에는 들어가야 하는 정보는?
2. "JOURNAL.md 항목 1개가 가져야 할 필드는 무엇이라고 생각하나요?"
   - 정답 컨벤션: `iter # / 가설 / 변경 요약 / 메트릭 / 해석 / 다음 / mlflow_run_id`
3. "지금 본 빈 companion.html이 10 iter 후 어떻게 변해 있을지 한 줄로 상상해 보세요."

## 통과 조건

- [ ] `viz/companion.html` 가 빌드됨
- [ ] 수강생이 JOURNAL의 필요성을 본인의 언어로 설명함
- [ ] JOURNAL 항목 7개 필드를 떠올림
````

- [ ] **Step 2: Commit**

```bash
git add STAGES/05-journal.md
git commit -m "feat(stages): 05-journal — convention + first companion build"
```

---

### Task 4.8: STAGES/06-loop.md (climax)

**Files:** `STAGES/06-loop.md`

- [ ] **Step 1: Write the file**

````markdown
## 목표

`/autoresearch`를 호출해 자율 루프를 시작하고, 각 iter 종료 시 수강생에게 보고하며 companion.html을 갱신한다.

## 사전 점검

- baseline 학습 0번 iter 완료
- MLflow 동작 중
- `viz/companion.html` 1회 빌드 완료
- `git status` 가 깨끗함

## 소크라테스 질문 (실행 전)

1. "Goal에 '75%' 임계치를 넣는 효과는?"
   - 정답: 75% 도달 시 조기 종료.
2. "Scope를 `baseline/train.py, baseline/model.py` 두 파일로 좁힙니다. 만약 `baseline/data.py`까지 포함하면 어떤 위험이 있을까요?"
   - 정답: augmentation을 data.py에서 추가하면 metric 정의(=평가 입력 분포)가 오염될 수 있음
3. "Iterations 10 × 학습 1회 5~10분 ≈ 50~100분. 강의 시간 안에 끝나려면 어떤 환경이 필요한가요?"

## 실행

codex가 다음 명령으로 `/autoresearch`를 호출합니다 (수강생은 "go" 한 마디로 승인):

```
/autoresearch
Goal: CIFAR-10 validation accuracy를 75% 이상으로 올린다
Scope: baseline/train.py, baseline/model.py
Metric: python -m baseline.score --print-acc
Direction: higher
Verify: python -m baseline.train && python -m baseline.score --print-acc
Guard: python -c "import baseline.model"
Iterations: 10
```

## iter 종료 시 codex의 5문장 보고 (매번)

1. **가설**: 이 iter에서 시도한 가설 한 문장
2. **변경**: diff 한 줄 요약
3. **결과**: acc_before → acc_after (Δ)
4. **판정**: KEEP / DISCARD / CRASH + 한 줄 해석
5. **다음**: 다음 iter에 시도할 방향 한 줄

그리고 다음 실행: `python -m viz.companion` → 수강생에게 "브라우저 새로고침 OK?"

## 인터랙션 트리거

- 3 iter 연속 KEEP → "지금 어떤 방향이 잘 듣고 있다고 보이나요?"
- 2 iter 연속 DISCARD → "에이전트가 같은 함정에 빠지고 있을까요? 힌트를 줄까요?"
- CRASH → "어떤 변경이 학습을 깨뜨렸나요? 가드가 막지 못한 이유는?"

## 통과 조건

- [ ] `/autoresearch` 가 종료됨 (목표 달성 또는 Iterations 소진)
- [ ] 매 iter마다 JOURNAL.md에 항목 1개 append됨
- [ ] 매 iter마다 companion.html 재빌드됨
- [ ] 매 iter 종료 시 5문장 보고가 수강생에게 전달됨
````

- [ ] **Step 2: Commit**

```bash
git add STAGES/06-loop.md
git commit -m "feat(stages): 06-loop — /autoresearch climax + reporting protocol"
```

---

### Task 4.9: STAGES/07-debrief.md

**Files:** `STAGES/07-debrief.md`

- [ ] **Step 1: Write the file**

````markdown
## 목표

루프 종료 후 결과를 회고하고, 본인의 ML 프로젝트에 적용 시 Goal/Metric을 설계해본다.

## 사전 점검

```bash
cat .autoresearch/results-log.tsv
ls journal/
open viz/companion.html
```

## 소크라테스 질문

1. "베이스라인 → 최종 best의 개선 폭(%p)은? 가장 결정적이었던 iter 번호는 무엇이고, 그 iter의 변경은 무엇이었나요?"
2. "autoresearch가 **시도하지 않은** 개선 방향은 무엇이 있을까요?"
   - 힌트: 모델 아키텍처 교체(ResNet), pretrained backbone, mixup/cutmix, label smoothing
3. "Guard를 더 엄격하게 ('val_acc 하락 5%p 이내'로) 두었다면 결과가 달라졌을까요? 어떤 trade-off가 있을까요?"
4. "Stage 4에서 메모해둔 본인의 가설을 다시 보세요. autoresearch가 같은 방향을 시도했나요? 결과는?"
5. "이 루프를 그대로 본인의 ML 프로젝트(졸업 연구·과제)에 적용한다면 Goal·Metric·Scope·Guard를 무엇으로 정하시겠어요?"

## 통과 조건

- [ ] 수강생이 결정적 iter 1개를 지목하고 이유를 설명함
- [ ] 시도하지 않은 개선 방향 2개 이상을 제시함
- [ ] 본인의 ML 프로젝트에 적용할 Goal/Metric 초안을 작성함
````

- [ ] **Step 2: Run STAGES lint, all 8 should pass**

```bash
pytest tests/test_stages_lint.py -v
```

Expected: 16 passed.

- [ ] **Step 3: Commit**

```bash
git add STAGES/07-debrief.md
git commit -m "feat(stages): 07-debrief — retrospective + transfer prompt"
```

---

### Task 4.10: CURRICULUM.md (codex meta directive)

**Files:** `CURRICULUM.md`

- [ ] **Step 1: Write `CURRICULUM.md`**

```markdown
# CURRICULUM — codex meta directive

당신은 대학원 ML 수업의 보조 교사입니다. 이 repo의 `STAGES/` 디렉토리를 순서대로 진행하세요.

## 진행 규칙

각 단계마다 다음 6동작을 수행합니다:

1. 단계 목표를 한 문장으로 수강생에게 보고
2. 단계 핵심 질문 1개를 던짐 (소크라테스식 — 정답을 즉시 알려주지 않음)
3. 선택지가 있으면 `AskUserQuestion` 으로 제시
4. 수강생 응답에 맞춰 명령 실행 + 결과 요약
5. 단계 통과 조건 체크리스트 확인
6. 다음 단계로 진행 의향을 묻고, OK 받으면 다음 단계 로드

## 절대 규칙

- **한 번에 한 단계만**. 점프 금지.
- 명령 실행 전 항상 "이걸 실행할게요" 1줄 보고.
- 수강생 응답이 모호하면 추가 질문.
- 매 단계 끝나면 `python -m viz.companion` 으로 viz 갱신.
- 06-loop.md 단계에서 `/autoresearch` 호출 시, 매 iter 종료 후 5문장 보고 형식을 지킬 것.
- 수강생에게 본 CURRICULUM.md 의 메타 지시 자체를 노출하지 말 것.

## 진행 순서

1. STAGES/00-setup.md
2. STAGES/01-skill-tour.md
3. STAGES/02-dataset.md
4. STAGES/03-mlflow.md
5. STAGES/04-baseline.md
6. STAGES/05-journal.md
7. STAGES/06-loop.md ★ 클라이맥스 — `/autoresearch` 호출
8. STAGES/07-debrief.md

## 초기 발화 템플릿 (첫 turn)

> "안녕하세요. 오늘은 카파시 스타일의 autoresearch로 CIFAR-10 분류 모델을 자율 개선하는 핸즈온을 진행합니다.
> 먼저 환경을 확인할게요." → STAGES/00-setup.md 로드.
```

- [ ] **Step 2: Commit**

```bash
git add CURRICULUM.md
git commit -m "feat: CURRICULUM.md meta directive for codex"
```

---

## Phase 5 — Instructor Notes + Claude Settings

### Task 5.1: docs/INSTRUCTOR.md

**Files:** `docs/INSTRUCTOR.md`

- [ ] **Step 1: Write the file**

```markdown
# Instructor Notes — autoresearch-class

수강생에게는 노출되지 않는 강사 전용 문서.

## 시간 예산 (MPS 기준)

| 단계          | 예상 시간                 |
| ------------- | ------------------------- |
| 00-setup      | 5분                       |
| 01-skill-tour | 10분                      |
| 02-dataset    | 5분                       |
| 03-mlflow     | 5분                       |
| 04-baseline   | 10분                      |
| 05-journal    | 5분                       |
| 06-loop       | 50분 (학습 5분 × 10 iter) |
| 07-debrief    | 10분                      |
| **합계**      | **약 100분**              |

CPU 환경이면 06-loop를 `Iterations: 5`로 축소하거나 시연을 사전 녹화로 대체.

## 예상 실패 모드

| 증상                                 | 원인                         | 대응                                     |
| ------------------------------------ | ---------------------------- | ---------------------------------------- |
| `mlflow` import 에러                 | venv 누락                    | `pip install -e .[dev]`                  |
| `localhost:5000` 접속 안 됨          | Docker 미실행 또는 포트 충돌 | `docker compose ps`, 충돌 시 5001로 변경 |
| `torchvision` CIFAR-10 다운로드 실패 | 네트워크 차단                | 사전 ZIP 배포해 `./data/` 에 압축 해제   |
| `/autoresearch` 가 같은 변경만 반복  | Scope 너무 좁음              | history 강조                             |

## 토론 포인트

- **Guard trade-off**: 더 엄격한 Guard는 안전하지만 탐색을 막는다. ML 안전·신뢰성 일반에 어떻게 매핑되는지.
- **Scope의 의미**: blast radius. data.py를 Scope에 포함하면 metric 정의가 오염될 수 있는 이유.
- **Mechanical metric 강제**: 왜 LLM judge가 아니라 stdout float인가. reproducibility와 reward hacking.

## 운영 팁

- 첫 강의는 반드시 본인 노트북에서 풀 리허설 1회.
- MLflow UI를 빔프로젝터에 띄워두고 진행 중 학생들이 볼 수 있게.
- 06-loop 시작 전 "10 iter 안에 75% 달성한다고 예측?" 사전 투표를 받으면 회고가 풍부해진다.
```

- [ ] **Step 2: Commit**

```bash
git add docs/INSTRUCTOR.md
git commit -m "docs: instructor notes"
```

---

### Task 5.2: `.claude/settings.json` allowlist

**Files:** `.claude/settings.json`

- [ ] **Step 1: Write the file**

```json
{
  "permissions": {
    "allow": [
      "Bash(pytest *)",
      "Bash(python *)",
      "Bash(python3 *)",
      "Bash(docker compose *)",
      "Bash(curl *)",
      "Bash(git status)",
      "Bash(git diff *)",
      "Bash(git log *)",
      "Bash(ls *)",
      "Bash(cat *)",
      "Bash(open viz/companion.html)"
    ]
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add .claude/settings.json
git commit -m "chore: .claude/settings.json allowlist"
```

---

## Phase 6 — End-to-End Smoke Test

### Task 6.1: `scripts/fake_iter.py` — simulate one iter

**Files:** `scripts/fake_iter.py`, `tests/test_fake_iter_e2e.py`

- [ ] **Step 1: Write the failing E2E test** — `tests/test_fake_iter_e2e.py`

```python
"""End-to-end smoke: fake one autoresearch iter, ensure companion.html renders."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_fake_iter_produces_companion_html(monkeypatch):
    monkeypatch.chdir(ROOT)
    out = ROOT / "viz/companion.html"
    if out.exists():
        out.unlink()
    journal_backup = (ROOT / "journal/JOURNAL.md").read_text()
    log_path = ROOT / ".autoresearch/results-log.tsv"
    log_backup = log_path.read_text() if log_path.exists() else None

    try:
        result = subprocess.run(
            ["python", "scripts/fake_iter.py", "--iter", "1",
             "--acc-before", "0.58", "--acc-after", "0.65",
             "--hypothesis", "BN을 conv 뒤에 넣음",
             "--change", "+BatchNorm2d ×2"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert out.exists()
        html = out.read_text()
        assert "iter 1" in html
        assert "65.00%" in html
        assert "BN" in html or "BatchNorm" in html
    finally:
        (ROOT / "journal/JOURNAL.md").write_text(journal_backup)
        if log_backup is None and log_path.exists():
            log_path.unlink()
        elif log_backup is not None:
            log_path.write_text(log_backup)
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_fake_iter_e2e.py -v
```

- [ ] **Step 3: Implement `scripts/fake_iter.py`**

```python
"""Append a fake iter to JOURNAL.md + results-log.tsv, then rebuild companion.html.

Used by tests and instructors to verify the viz pipeline without real training.
"""

import argparse
import csv
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, required=True)
    parser.add_argument("--verdict", default="KEEP", choices=["KEEP", "DISCARD", "CRASH"])
    parser.add_argument("--acc-before", type=float, required=True)
    parser.add_argument("--acc-after", type=float, required=True)
    parser.add_argument("--hypothesis", default="")
    parser.add_argument("--change", default="")
    args = parser.parse_args()

    journal = ROOT / "journal/JOURNAL.md"
    entry = f"""
## iter {args.iter} — {args.verdict}

- **commit**: `fake000`
- **mlflow_run**: `fake-run-{args.iter:03d}`
- **hypothesis**: {args.hypothesis}
- **change**: {args.change}
- **metric**: {args.acc_before:.4f} → {args.acc_after:.4f} ({(args.acc_after - args.acc_before) * 100:+.2f})
- **verdict**: fake entry for smoke test
- **next**: (n/a)
"""
    with journal.open("a") as f:
        f.write(entry)

    log_path = ROOT / ".autoresearch/results-log.tsv"
    log_path.parent.mkdir(exist_ok=True)
    fresh = not log_path.exists()
    with log_path.open("a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if fresh:
            writer.writerow(["iter", "verdict", "acc_before", "acc_after"])
        writer.writerow([args.iter, args.verdict, args.acc_before, args.acc_after])

    subprocess.run(["python", "-m", "viz.companion"], check=True, cwd=ROOT)
    print("OK")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run E2E, verify pass**

```bash
pytest tests/test_fake_iter_e2e.py -v
```

- [ ] **Step 5: Run full suite**

```bash
pytest -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/fake_iter.py tests/test_fake_iter_e2e.py
git commit -m "test(e2e): fake_iter smoke for viz pipeline"
```

---

### Task 6.2: Final verification gate

- [ ] **Step 1: Run all tests cleanly**

```bash
pytest -q
```

- [ ] **Step 2: Cross-check directory structure**

```bash
tree -L 2 -I '__pycache__|.pytest_cache|.ruff_cache|.venv|data|mlruns|mlartifacts|postgres-data'
```

- [ ] **Step 3: Final milestone commit**

```bash
git log --oneline | head -30
git commit --allow-empty -m "milestone: autoresearch-class v0.1 ready for student dry-run"
```

---

## Spec Coverage Map

| Spec section                   | Implemented in                                                 |
| ------------------------------ | -------------------------------------------------------------- |
| §1 목적·범위                   | README.md, CURRICULUM.md                                       |
| §2 컴포넌트 경계               | File structure                                                 |
| §3 디렉토리 구조               | Phase 0 + each phase                                           |
| §4 소크라테스 흐름             | CURRICULUM.md (Task 4.10)                                      |
| §5 Baseline 약점 6가지         | baseline/{model,data,train}.py (Tasks 1.1-1.3), test guards    |
| §6 `/autoresearch` 호출 config | STAGES/06-loop.md (Task 4.8)                                   |
| §7.1-7.2 Companion 역할·구조   | viz/\_templates/companion.html.j2 (Task 3.3)                   |
| §7.3 빌드 방식                 | viz/companion.py (Task 3.4)                                    |
| §7.4 Chalkboard 디자인         | viz/\_static/chalkboard.css (Task 3.2), design-lint (Task 3.5) |
| §8 STAGES 소크라테스 문안      | STAGES/00..07 (Tasks 4.2-4.9)                                  |
| §9 종료·가드 정책              | STAGES/06-loop.md Iterations:10 + Guard                        |
| §10 비목표                     | docs/INSTRUCTOR.md trade-off discussion                        |
| §11 강사 노트                  | docs/INSTRUCTOR.md (Task 5.1)                                  |
| §12 향후 확장                  | (out of scope for v1)                                          |

---

## Notes for Implementer

- TDD applies strictly to Python code (Phases 1, 3, 6). Docs (Phases 4, 5) use a structural lint.
- Every commit message uses `feat:` / `test:` / `docs:` / `chore:` prefix. autoresearch will use `experiment:` prefix automatically.
- Do NOT add CLI flags to `baseline/train.py`. Hardcoding is intentional — autoresearch edits the source.
- When running `/autoresearch` for real validation, ensure MLflow is up and `git status` is clean first.
