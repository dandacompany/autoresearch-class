# autoresearch-class

대학원 머신러닝 강의용 핸즈온: 카파시 스타일 autoresearch로 CIFAR-10 모델을 자율 개선합니다.

## Quick start

```bash
git clone <this-repo> autoresearch-class
cd autoresearch-class

# 환경 셋업 (1회)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# MLflow 기동 (macOS에서 포트 5000이 ControlCenter와 충돌하면 MLFLOW_HOST_PORT=5001 prefix)
MLFLOW_HOST_PORT=${MLFLOW_HOST_PORT:-5000} docker compose -f mlflow/docker-compose.yml up -d

# codex 진입
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

## 트러블슈팅

- **포트 5000 충돌 (macOS)**: AirPlay Receiver가 5000을 점유합니다. `MLFLOW_HOST_PORT=5001 docker compose ...` 으로 우회하고, 학습 스크립트에는 `MLFLOW_TRACKING_URI=http://127.0.0.1:5001`을 export.
- **`pip install -e .` 실패**: pyproject 패치 적용 전 fresh clone에서는 setuptools가 STAGES/data/journal/mlflow를 패키지로 인식. 최신 `main` 브랜치로 pull 후 재시도.
