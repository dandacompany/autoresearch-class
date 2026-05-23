# TUTORIAL — autoresearch-class 한 번에 따라하기

대학원 ML 수업 핸즈온 가이드. codex CLI에서 `/autoresearch`로 CIFAR-10 모델을 자율 개선하는 전 과정을 단계별 프롬프트와 함께 정리했습니다.

소요 시간 (MPS 기준): **약 100분** (CPU 환경은 150~180분).

---

## Part 0 — 사전 준비 (수업 시작 전 30분)

수업 당일 시작 직전에 점검해야 할 항목입니다.

### 0.1 codex CLI 설치 + OAuth 인증

```bash
# codex CLI가 없으면 설치 (이미 있으면 스킵)
brew install codex   # 또는 vendor 권장 방법

# OAuth 로그인
codex auth login
codex --version      # 0.130+ 권장
```

### 0.2 Docker Desktop 기동

학습용 MLflow 서버를 docker-compose로 띄웁니다. 사전에 Docker Desktop을 켜두세요.

```bash
docker version --format '{{.Server.Version}}'
```

### 0.3 Python 3.11+

```bash
python3.11 --version || python3 --version
# 없으면 brew install python@3.11 (macOS)
```

### 0.4 `cods` alias (선택, 권장)

YOLO 모드 단축. `~/.zshrc` 또는 `~/.bashrc`에 추가:

```bash
alias cods='codex --dangerously-bypass-approvals-and-sandbox'
```

YOLO 모드는 수업 데모용입니다. 수업이 끝나면 `cods` 대신 일반 `codex`로 돌아가세요.

---

## Part 1 — 리포 클론과 환경 셋업

### 1.1 클론

```bash
git clone https://github.com/dandacompany/autoresearch-class.git
cd autoresearch-class
```

### 1.2 가상환경 + 의존성

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

설치는 약 1~2분 소요됩니다 (torch ~700MB). 마지막에 다음이 출력되면 성공:

```
Successfully installed autoresearch-class-0.1.0 ...
```

### 1.3 MLflow 서버 기동

```bash
# 기본 포트 (5000)
docker compose -f mlflow/docker-compose.yml up -d
```

**⚠️ macOS 사용자 주의**: 포트 5000은 ControlCenter(AirPlay Receiver)가 점유합니다. 충돌 시:

```bash
MLFLOW_HOST_PORT=5001 docker compose -f mlflow/docker-compose.yml up -d
```

검증:

```bash
# 기본 포트: 5000, 우회: 5001
curl -fsS http://127.0.0.1:5001/health   # → OK
```

처음 기동 시 mlflow 이미지가 psycopg2-binary를 자동 설치하므로 15~30초 정도 걸립니다.

### 1.4 codex 진입 + 스킬 확인

```bash
cods
# 또는 codex --dangerously-bypass-approvals-and-sandbox
```

처음 진입 시 codex가 다음을 묻습니다:

- "Trust this directory?" → **1. Yes, continue**
- "Migrate external agent configs?" → **2. Skip for now** (개인 codex 설정 보호)

codex 프롬프트에 진입했으면 다음 한 줄을 입력:

```
@CURRICULUM.md 대로 진행해줘
```

이후 codex가 단계별로 안내합니다.

---

## Part 2 — Stage별 진행 가이드

각 stage 끝에는 **codex가 "다음 단계로 진행할까요? (y/n)"** 형태로 묻습니다. 준비됐으면 `y`, 아직 질문이 남았으면 자유롭게 추가 질문하세요.

### Stage 00 — 환경 점검

codex가 자동 실행:

- `codex --version`, `docker version`, `python3.11 --version`
- autoresearch 스킬 확인 (`~/.agents/skills/autoresearch` 또는 codex plugin marketplace)

**핵심 질문**: "왜 단순 학습 스크립트보다 자율 루프가 필요한가?"

> **모범 답안 예시**: 학습 스크립트 1회는 한 가설만 검증하지만, 자율 루프는 가설→실험→해석을 반복하면서 git 메모리에 기록하니까 사람 연구자처럼 누적 학습이 가능합니다.

**선택**:

- 진행 모드 1 (codex가 명령 자동 실행) — **권장**
- 진행 모드 2 (학생이 직접 명령 입력)

### Stage 01 — autoresearch 스킬 투어

소크라테스 질문 4개:

**Q1. Metric이 단일 float이어야 하는 이유?**

> 단일 float이어야 루프 엔진이 직전 값과 기계적으로 비교해 keep/discard를 판정할 수 있습니다. JSON·다중라인이면 파싱 오류·필드명 변경에 취약하고, LLM이 재해석해야 해서 reward hacking 위험도 커집니다.

**Q2. 한 iter에 한 변경만 하는 이유?**

> 두 변경을 동시에 하면 성능 변화의 인과를 분리할 수 없습니다. 좋아져도 누구 덕인지 모르고, 나빠져도 누구를 되돌려야 할지 애매해서 다음 iter의 가설 좁히기가 어렵고 rollback이 무딘 도구가 됩니다.

**Q3. Guard와 Metric의 차이?**

> Guard는 어떤 iter도 어겨선 안 되는 invariant 게이트(boolean), Metric은 iter마다 더 나아져야 하는 측정값(비교 가능한 숫자)입니다.

**Q4. "git 커밋이 곧 메모리"의 의미?**

> 각 iter 변경이 커밋되면 git log 자체가 실험 history이자 메모리입니다. 커밋 없이 바꾸면 (a) rollback 단위가 사라지고 (b) results-log의 metric과 코드 상태가 매칭되지 않으며 (c) 새 iter가 이전 실험을 덮어쓰면 가설 비교가 깨집니다.

stage 종료 시 codex가 `python -m viz.companion`을 실행해 viz/companion.html을 빌드합니다. **브라우저에서 열어보세요** — 아직 비어있지만 디자인이 적용되어 있어야 합니다.

### Stage 02 — CIFAR-10 데이터셋

codex가 자동 실행:

- `python -c "from baseline.data import get_loaders; ..."` (CIFAR-10 다운로드 ~170MB, 11초)
- `pytest tests/test_data.py -v` (2 passed)

**Q1. `./data/` git 처리는?**

> `.gitignore`에 추가. Git LFS는 오버킬, 외부 캐시 디렉토리는 학생 환경 불일치 위험.

**Q2. Sanity check 3가지?**

> 클래스 분포 (10 클래스 균등?) / 이미지 shape (3×32×32) / 정규화 통계 (mean/std 매핑 확인).

**Q3. 왜 augmentation 가드 테스트를 두는가?**

> baseline에 augmentation이 이미 있으면 autoresearch가 "RandomFlip 추가" 같은 명백한 첫 가설을 못 씁니다. 가드 테스트는 미래의 누군가가 무심결에 augmentation을 넣어버려서 교육 효과를 망치는 것을 막습니다.

### Stage 03 — MLflow 서버 점검

Part 1.3에서 이미 띄웠으면 codex는 health check만 합니다.

**Q1. mlflow ui (단일 프로세스) vs docker-compose self-host?**

> docker-compose는 postgres에 메타스토어가 영구 저장돼 재기동 후에도 run 이력이 보존되고, 동일 이미지로 학생 환경 재현성도 확보됩니다. mlflow ui는 파일시스템 의존적이라 위치/권한 차이로 학생마다 다른 결과를 만들 위험이 있습니다.

**Q2. MLflow의 experiment vs run?**

> experiment = 연구 주제(같은 목표를 추구하는 run들의 묶음), run = 1회 학습. autoresearch 1회 호출은 1 experiment (이름: `autoresearch-class`), N runs (iter별 1 run)에 매핑됩니다.

**Q3. 매 iter 후 MLflow UI에서 볼 것?**

> Compare 기능으로 여러 run의 val_acc 라인 차트 겹쳐 보기, params diff (어떤 하이퍼파라미터가 바뀌었는지), artifacts (model.py 스냅샷으로 정확히 어떤 코드를 돌렸는지).

### Stage 04 — Baseline 학습 (★ 의도적 약점 체험)

```bash
# codex가 자동 실행 — 학생은 OK만
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001   # 포트 5001 사용 중일 때
python -m baseline.train
```

학습 시간: MPS 5분 / CPU 10분 / NVIDIA GPU 1분.

**Q1. baseline이 가지지 말아야 할 것은? (다중)**

> ① BatchNorm ② Augmentation ③ LR scheduler ④ Dropout ⑤ Adam — **다섯 가지 모두**. 각각이 autoresearch가 발견할 명백한 첫 가설입니다.

**Q2. 최종 val_acc는?**

> 예상: 55~62% (베이스라인의 약점 6가지 때문). 이 숫자가 autoresearch의 0번 iter 기준점이 됩니다.

**Q3. MLflow UI에서 본 params·metrics는?**

> http://127.0.0.1:5001 접속 → autoresearch-class experiment → 방금 run 클릭 → Parameters 탭(model/optimizer/lr/batch_size/epochs), Metrics 탭(train_loss/val_loss/val_acc).

**Q4. 75%로 끌어올리려면 본인은 무엇을 바꿀까?**

> **이 답을 메모해두세요**. Stage 07 회고에서 autoresearch의 시도와 비교합니다.

### Stage 05 — Journal 컨벤션 + companion.html

```bash
cat journal/TEMPLATE.md      # 항목 형식 확인
python -m viz.companion      # 빈 페이지 빌드
open viz/companion.html      # 브라우저에서 열기
```

**Q1. tsv가 있는데 JOURNAL.md를 별도로 두는 이유?**

> tsv는 숫자만, JOURNAL은 **가설·해석·다음 시도** 같은 서사. 사람이 읽으면서 "왜 그 변경이 좋았는지"를 추적하기 위함.

**Q2. JOURNAL 항목 1개의 필드?**

> `iter # / 가설 / 변경 요약 / 메트릭(before→after, Δ) / 해석 / 다음 시도 / mlflow_run_id` (7개 필드).

**Q3. 빈 companion.html이 10 iter 후 어떻게 변할까?**

> 상단에 best metric 큰 숫자, 메트릭 라인차트, iter별 카드 그리드. 각 카드는 KEEP/DISCARD/CRASH 배지와 함께 회전 각도가 살짝 어긋난 분필 손글씨 톤.

### Stage 06 — `/autoresearch` 자율 루프 ★ 클라이맥스

**사전 점검**:

- baseline 학습 0번 iter 완료 (`.autoresearch/last_val_acc.txt` 존재)
- MLflow 동작 중
- `git status` 깨끗

**실행 전 소크라테스 3문항**:

**Q1. Goal에 "75%" 임계치를 넣는 효과?**

> 75% 도달 시 조기 종료. Iterations 10과 함께 2가지 종료 조건이 동시에 작동.

**Q2. Scope에 data.py 포함 시 위험?**

> data.py에서 augmentation을 추가하면 metric 정의(=평가 입력 분포)가 오염됩니다. 학습 데이터에는 augmentation이 OK지만 val/test에도 적용되면 평가가 변형돼서 같은 acc 숫자가 다른 의미를 갖게 됩니다.

**Q3. Iterations 10 × 학습 5~10분 = 50~100분 시간 예산. 어떤 환경 필요?**

> 강의 시간 안에 끝나려면 MPS 또는 NVIDIA GPU. CPU만 있으면 Iterations를 5로 줄이거나 사전 녹화로 대체.

**호출 명령** (codex가 자동 실행):

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

**매 iter 종료 시 codex의 5문장 보고**:

1. **가설**: 이 iter에서 시도한 가설 한 문장
2. **변경**: diff 한 줄 요약
3. **결과**: acc_before → acc_after (Δ)
4. **판정**: KEEP / DISCARD / CRASH + 한 줄 해석
5. **다음**: 다음 iter에 시도할 방향

그리고 `python -m viz.companion` 재빌드 → "브라우저 새로고침 OK?" 안내.

**인터랙션 트리거** (codex가 학생에게 묻는 시점):

- **3 iter 연속 KEEP**: "지금 어떤 방향이 잘 듣고 있다고 보이나요?"
- **2 iter 연속 DISCARD**: "에이전트가 같은 함정에 빠지고 있을까요? 힌트를 줄까요?"
- **CRASH**: "어떤 변경이 학습을 깨뜨렸나요? Guard가 막지 못한 이유는?"

**예상 궤적**:

- iter 1-2: BatchNorm 추가 → val_acc 58 → 65~68%
- iter 2-3: LR scheduler (cosine/step) → +2~4%p
- iter 3-4: Data augmentation (RandomFlip/Crop) → +4~6%p
- iter 4-5: Optimizer SGD → SGD+momentum 또는 Adam → +1~3%p
- iter 6-7: epoch 5 → 10~15 → +2~5%p
- iter 8-10: 마이크로 튜닝 (dropout, channel width) → +1~3%p

목표 75% 도달하면 조기 종료. 미도달 시 Iterations 10 소진.

### Stage 07 — 회고

```bash
cat .autoresearch/results-log.tsv
ls journal/
open viz/companion.html
```

**Q1. 베이스라인 → 최종 best 개선 폭? 결정적이었던 iter는?**

> 보통 BatchNorm 추가(iter 1-2)가 가장 큰 폭. 카드 그리드의 Δ 값으로 즉시 확인 가능.

**Q2. autoresearch가 시도하지 않은 개선 방향?**

> 모델 아키텍처 교체(ResNet-mini), pretrained backbone, mixup/cutmix, label smoothing, EMA, distillation 등. 한 iter 한 변경 원칙 때문에 큰 아키텍처 변경은 잘 안 시도됨.

**Q3. Guard를 "val_acc 하락 5%p 이내"로 두면?**

> 안전성↑(망가뜨리는 변경 차단) 탐색력↓(과감한 시도 막힘). 학습 초반 정체 가능. **trade-off 관점에서 ML 안전·신뢰성 일반에 어떻게 매핑되는지**를 토론.

**Q4. Stage 04에서 메모해둔 본인 가설과 비교**:

> autoresearch가 같은 방향을 시도했나요? 결과는?

**Q5. 본인 ML 프로젝트에 적용한다면 Goal/Metric/Scope/Guard는?**

> 다음 과제로 가져갈 수 있는 transfer prompt. 본인 데이터·문제·평가 지표로 동일 구조 설계.

---

## Part 3 — 트러블슈팅

### 자주 발생하는 5가지 friction (시뮬레이션에서 발견)

| #   | 증상                                                                  | 원인                                     | 해결                                                                                                                                                                       |
| --- | --------------------------------------------------------------------- | ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `skills add anthropics/superpowers@autoresearch` GitHub 인증 실패     | 패키지가 private/잘못된 owner            | `skills find autoresearch` 로 marketplace 검색 후 실제 owner/repo 사용. 또는 codex plugin marketplace에 이미 등록되어 있는지 확인 (`/autoresearch:autoresearch` 호출 시도) |
| 2   | `pip install -e ".[dev]"` setuptools가 STAGES/data 등을 패키지로 인식 | flat-layout 자동 감지                    | 최신 main에는 `[tool.setuptools.packages.find] include = ["baseline*", "viz*"]` 포함됨. `git pull` 후 재시도                                                               |
| 3   | MLflow 컨테이너가 `http://127.0.0.1:5000/health`에 응답 안 함         | macOS ControlCenter가 5000 점유          | `MLFLOW_HOST_PORT=5001 docker compose -f mlflow/docker-compose.yml up -d`, 학습 시 `export MLFLOW_TRACKING_URI=http://127.0.0.1:5001`                                      |
| 4   | MLflow 컨테이너 로그에 `No module named 'psycopg2'`                   | mlflow 이미지가 postgres 드라이버 미포함 | 최신 docker-compose.yml은 컨테이너 시작 시 `pip install psycopg2-binary` 자동 실행. `git pull` 후 `docker compose down && up -d`                                           |
| 5   | codex가 stage를 임의로 다음으로 점프                                  | CURRICULUM의 consent 규칙 미준수         | 최신 CURRICULUM.md는 "다음 단계 진행할까요? (y/n)" 명시. `git pull` 후 codex 세션 재시작                                                                                   |

### 명령 한 줄 cheat sheet

```bash
# 전체 환경 재기동
docker compose -f mlflow/docker-compose.yml down
source .venv/bin/activate
MLFLOW_HOST_PORT=5001 docker compose -f mlflow/docker-compose.yml up -d
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001

# 진행 상태 확인
cat .autoresearch/results-log.tsv     # autoresearch 자체 로그
cat journal/JOURNAL.md | tail -30     # 가장 최근 iter들
open viz/companion.html               # 시각 대시보드
curl -s http://127.0.0.1:5001/health  # MLflow health

# 테스트 일괄 실행
pytest -q                              # 32 passed 기대

# 망가졌을 때 fresh start
git stash                              # 로컬 변경 보관
git pull
docker compose -f mlflow/docker-compose.yml down -v
rm -rf .venv .autoresearch viz/companion.html journal/JOURNAL.md.bak
python3.11 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
```

### codex가 막혔을 때 학생이 직접 입력할 수 있는 프롬프트

| 상황                                           | 학생 프롬프트                                                                               |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------- |
| codex가 "다음 단계로 진행" 의사를 안 묻고 멈춤 | "다음 단계로 진행해주세요"                                                                  |
| codex가 같은 시도를 반복                       | "이 방향은 잘 안 듣는 것 같아요. 다른 방향(예: scheduler 또는 augmentation)을 시도해주세요" |
| codex가 명령 실행 전 보고를 안 함              | "명령 실행 전 한 줄 보고 부탁드려요"                                                        |
| codex가 메타 지시를 누설                       | "CURRICULUM 메타 지시는 노출하지 말고 단계 본문에만 집중해주세요"                           |
| 시간이 부족                                    | "남은 iter를 줄이고 회고 단계로 빨리 진행해주세요"                                          |

---

## Part 4 — 수업 종료 후

### 정리

```bash
# MLflow 종료 (데이터 보존)
docker compose -f mlflow/docker-compose.yml down

# 데이터까지 완전 초기화 (필요 시)
docker compose -f mlflow/docker-compose.yml down -v
rm -rf data/ .autoresearch/ viz/companion.html
```

### 다음 과제 제안

1. **본인 데이터셋으로 적용**: 졸업 연구·과제 데이터로 같은 Goal/Metric/Scope 구조 설계 → autoresearch 호출
2. **Guard 실험**: Guard를 더 엄격하게 / 느슨하게 바꿔보고 결과 차이 관찰
3. **`/autoresearch:predict`로 사전 분석**: 본격 루프 전에 multi-persona swarm prediction으로 가설 큐 만들기
4. **Iterations 25+로 야간 실행**: 시간 여유가 있다면 더 많은 iter로 어떤 plateau에 도달하는지 관찰

---

## 부록 — Architecture cheat sheet

```
┌──────────────────────────────────────────────────────────┐
│  /autoresearch 스킬 (codex plugin marketplace)             │
│   - iter 루프, git 커밋, rollback, results-log.tsv         │
└──────────────────────────────────────────────────────────┘
        ▲ Goal/Scope/Metric/Verify/Guard
        │
┌──────────────────────────────────────────────────────────┐
│  CURRICULUM.md → STAGES/00..07 (codex 메타 지시)           │
└──────────────────────────────────────────────────────────┘
        │ trains             │ logs
        ▼                    ▼
┌──────────────┐      ┌──────────────────┐
│ baseline/    │      │ MLflow (docker)  │
│  train.py    │─────▶│  UI :5001        │
│  model.py    │      │                  │
│  data.py     │      └──────────────────┘
│  score.py    │
└──────────────┘
        │ append
        ▼
┌──────────────────┐      ┌──────────────────────┐
│ journal/         │─────▶│ viz/companion.html   │
│  JOURNAL.md      │      │ (Chalkboard 디자인)   │
└──────────────────┘      └──────────────────────┘
```

각 단위 책임:

- **`/autoresearch`**: 의사결정·코드 패치·루프 제어. 학생이 직접 수정 X
- **CURRICULUM + STAGES**: codex가 읽고 학생을 안내. 학생은 안 봐도 됨
- **baseline/**: 의도적으로 약한 학습 코드. autoresearch의 수정 대상
- **mlflow/**: 메트릭·아티팩트 저장소
- **journal/**: 사람이 읽는 서사 노트
- **viz/**: 정적 HTML 대시보드. iter마다 재빌드

---

## 부록 — 시간 예산 (실측 + 예측)

| 단계     | 활동                             | MPS  | CPU   | 누적 (MPS) |
| -------- | -------------------------------- | ---- | ----- | ---------- |
| Part 0   | 사전 준비 (codex/docker/python)  | 5분  | 5분   | 5분        |
| Part 1   | clone + venv install + mlflow up | 5분  | 5분   | 10분       |
| Stage 00 | 환경 점검 + 모드 결정            | 5분  | 5분   | 15분       |
| Stage 01 | 스킬 투어 (4문항)                | 10분 | 10분  | 25분       |
| Stage 02 | CIFAR-10 다운로드 + sanity       | 5분  | 5분   | 30분       |
| Stage 03 | MLflow health check              | 3분  | 3분   | 33분       |
| Stage 04 | baseline 1회 학습                | 8분  | 12분  | 41분       |
| Stage 05 | journal + companion              | 5분  | 5분   | 46분       |
| Stage 06 | autoresearch loop (10 iter)      | 50분 | 100분 | 96분       |
| Stage 07 | 회고 + transfer prompt           | 10분 | 10분  | 106분      |

CPU 환경에서는 Stage 06을 `Iterations: 5`로 축소 권장 (총 시간 75분으로 단축).

---

## License

이 가이드는 `dandacompany/autoresearch-class` repo의 일부로 MIT License 하에 자유롭게 사용 가능합니다. 강의 자료로 활용 시 출처 표기 환영합니다.

---

**문서 버전**: v0.3 (실제 시뮬레이션 기반)
**작성**: 2026-05-23
**채널**: [Dante Labs Discord](https://discord.com/invite/rXyy5e9ujs)
