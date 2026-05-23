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
