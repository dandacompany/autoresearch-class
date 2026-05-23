# autoresearch-class — 대학원 ML 강의용 교육 repo 설계서

- **작성일**: 2026-05-23
- **상태**: brainstorming → design approved
- **다음 단계**: writing-plans 로 구현 계획 수립
- **저장소 이름(예정)**: `autoresearch-class`
- **타깃 수강생**: 국민대학교 경영대학원 AI빅데이터MBA 등 대학원생

---

## 1. 목적과 범위

수강생이 codex CLI 위에서 `/autoresearch` 스킬을 호출해 **약한 CIFAR-10 분류 베이스라인을 자율 루프로 개선**하는 과정을 1시간 분량의 핸즈온으로 체험하게 한다. 카파시 autoresearch의 핵심 정신(Modify → Verify → Keep/Discard → Repeat, git이 곧 메모리, mechanical verification)을 **수강생이 직접 손으로 돌리며** 이해하는 것이 목표.

**비목표**: 새로운 autoresearch 엔진 구현, AutoML 도구 비교 연구, GPU 대규모 학습.

## 2. 컴포넌트 경계

`/autoresearch` 스킬이 루프 엔진을 전담한다 (이미 git 커밋·rollback·results-log·iterations 상한 모두 제공). 본 repo는 **스킬이 잘 돌도록 환경을 깔고, 소크라테스식으로 진행을 안내하며, 결과를 시각화**하는 단위들을 제공한다.

```
┌──────────────────────────────────────────────────────────┐
│  /autoresearch 스킬 (외부 — 이미 존재)                     │
│   - iter 루프, git 커밋, rollback, results-log.tsv         │
└──────────────────────────────────────────────────────────┘
        ▲ Goal/Scope/Metric/Verify/Guard
        │
┌──────────────────────────────────────────────────────────┐
│  CURRICULUM.md (codex 메타 지시)                          │
│  └─ STAGES/00..07 (단계 본문, 수강생 학습 흐름)             │
└──────────────────────────────────────────────────────────┘
        │ trains             │ logs
        ▼                    ▼
┌──────────────┐      ┌──────────────────┐
│ baseline/    │      │ MLflow (docker)  │
│  train.py    │─────▶│  UI :5000        │
│  model.py    │      │                  │
│  data.py     │      └──────────────────┘
│  score.py     │
└──────────────┘
        │ append
        ▼
┌──────────────────┐      ┌──────────────────────┐
│ journal/         │─────▶│ viz/companion.html   │
│  JOURNAL.md      │      │ (Chalkboard 디자인)   │
└──────────────────┘      └──────────────────────┘
```

각 단위 단독 책임:

- `/autoresearch` 스킬: 의사결정 + 코드 패치 + 루프 제어. 본 repo가 **수정하지 않는다**.
- CURRICULUM.md + STAGES/: codex가 읽고 수강생을 단계별로 안내하는 가이드. 수강생은 STAGES를 직접 읽지 않아도 됨.
- baseline/: 의도적으로 약점이 남은 학습 코드. autoresearch의 Scope 대상.
- mlflow/: 메트릭·아티팩트 저장소 + UI. 단방향 (쓰기만).
- journal/: 사람이 읽는 서사 노트. 각 iter 후 codex가 append.
- viz/: 정적 HTML 대시보드. 매 iter 후 codex가 재빌드.
- docs/INSTRUCTOR.md: 강사 전용. 수강생 흐름에는 노출되지 않음.

## 3. 디렉토리 구조

```
autoresearch-class/
├── README.md
├── CURRICULUM.md
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
│   ├── train.py
│   ├── model.py
│   ├── data.py
│   └── score.py
├── mlflow/
│   ├── docker-compose.yml
│   └── README.md
├── journal/
│   ├── JOURNAL.md          # 빈 템플릿
│   └── TEMPLATE.md
├── viz/
│   ├── companion.html      # 정적 산출물
│   ├── companion.py        # 빌더
│   └── assets/
├── .claude/
│   └── settings.json
└── docs/
    └── INSTRUCTOR.md
```

## 4. 소크라테스 흐름 (CURRICULUM.md)

CURRICULUM.md는 codex에게 주는 메타 지시문이다. 수강생 경험:

1. `git clone … && cd autoresearch-class`
2. `codex` 실행
3. 첫 발화: "@CURRICULUM.md 대로 진행해줘"
4. codex가 STAGES/00부터 순차 진행, 각 단계마다 다음 6동작 수행:
   1. 단계 목표 한 문장 보고
   2. 핵심 소크라테스 질문 1개 (정답 직접 제공 금지)
   3. 필요 시 `AskUserQuestion` 으로 선택지 제시
   4. 수강생 응답에 맞춰 명령 실행 + 결과 요약
   5. 단계 완료 체크리스트 확인
   6. 다음 단계 진행 의향 묻기

규칙: 한 번에 한 단계, 점프 금지. 명령 실행 전 "이걸 실행할게요" 1줄 보고. 매 단계 끝나면 `viz/companion.html` 갱신.

## 5. Baseline 설계 (`baseline/train.py`)

수강생이 "에이전트가 무엇을 발견할 수 있는지" 체감하려면 베이스라인이 개선 여지를 명시적으로 비워둬야 한다.

**의도적 약점 6가지**:

| #   | 약점                           | 예상 발견 시점         | 기대 개선 폭 |
| --- | ------------------------------ | ---------------------- | ------------ |
| 1   | Conv2D 2층, normalization 없음 | iter 1-2               | +5~8%p       |
| 2   | LR 고정 0.01, scheduler 없음   | iter 2-3               | +2~4%p       |
| 3   | Data augmentation 없음         | iter 3-4               | +4~6%p       |
| 4   | Optimizer SGD without momentum | iter 4-5               | +1~3%p       |
| 5   | Batch size 16                  | 자주 시도, 결정적 아님 | —            |
| 6   | epoch 5 (학습 부족)            | 후반                   | +2~5%p       |

**기대 궤적**: baseline 약 58% → 10 iter 후 약 75~80% val acc.

**학습 시간 가정**: 1회 학습 약 5분(애플 실리콘 MPS) / 약 10분(CPU only) / 약 1분(NVIDIA GPU). 강의에서는 MPS 또는 CPU 기준으로 안내.

**구현 원칙**:

- CLI 인자·config 파일 없이 **하드코딩**. argparse 인터페이스가 있으면 수강생이 "그냥 튜닝이네"로 오해.
- `mlflow.start_run()` 으로 학습 wrap.
  - params: 모델 클래스명, optimizer, lr, batch_size, epochs (코드 변경 시 자동 다름)
  - metrics: train_loss, val_loss, val_acc (epoch별 + 최종)
  - artifacts: model.py 스냅샷, 평가 혼동행렬 PNG
  - tags: `experiment_iter`, `commit_sha`

**score.py** 는 `--print-acc` 플래그 시 단일 float을 stdout으로 출력 (autoresearch Metric 명령 요건).

## 6. `/autoresearch` 호출 config (STAGES/06-loop.md 클라이맥스)

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

설계 포인트:

- Scope는 2파일로 한정 → data.py/score.py 변형으로 metric 정의가 오염될 위험 차단.
- Verify는 학습+평가 묶음 → caching 함정 회피.
- Guard는 import smoke test → "코드가 아예 깨지는 변경"만 방어. 너무 강하면 에이전트 탐색을 막음.
- Iterations 10 × 학습 1회 5~10분 ≈ 50~100분 (MPS/CPU). NVIDIA GPU 환경에서는 약 10분.

## 7. Visual Companion (`viz/companion.html`)

### 7.1 역할

MLflow UI는 메트릭/run 위주여서 "에이전트의 사고 흐름"이 안 보인다. companion.html은 **숫자 + 서사 + 코드 변경**을 시간축에 정렬해 보여주는 단일 정적 HTML 대시보드.

### 7.2 화면 구조

```
┌────────────────────────────────────────────────────────┐
│ autoresearch progress         iter 4/10 │ best 71.3%   │
├────────────────────────────────────────────────────────┤
│ [메트릭 라인차트]   [성공/실패 타임라인]                │
├────────────────────────────────────────────────────────┤
│ iter 카드 그리드 (iter별 1장)                           │
│  - 가설 / 변경 요약 / 결과 / 해석                       │
│  - [view git diff] [view mlflow run]                   │
└────────────────────────────────────────────────────────┘
```

### 7.3 빌드 방식

`viz/companion.py` 가 다음 셋을 읽어 HTML에 정적 임베드:

- `.autoresearch/results-log.tsv` (스킬 자동 생성)
- `journal/JOURNAL.md`
- `git log --grep "experiment:"`

매 iter 종료 시 codex가 `python viz/companion.py` 1회 실행 → 수강생에게 "브라우저 새로고침" 안내. 서버·웹소켓 없음.

### 7.4 디자인 시스템 — Dante Labs Chalkboard

surface=slides 시스템을 웹 페이지에 적응. 다크 슬레이트 칠판 + 분필 손글씨 톤.

**컬러 토큰 (CSS variables)**:

```css
:root {
  --bg-chalkboard: #2f4f4f;
  --bg-chalkboard-deep: #243f3f;
  --bg-chalkboard-raised: #3d5c5c;

  --ink-primary: #ffffff;
  --ink-muted: #d8d5c9;
  --ink-subtle: #9bafaf;

  --accent-tomato: #ff6347; /* 메타 요소만 */
  --accent-yellow: #ffd700; /* 본문 강조 */

  --stroke-chalk: rgba(255, 255, 255, 0.72);
  --stroke-chalk-soft: rgba(255, 255, 255, 0.36);
  --dust-chalk: rgba(255, 255, 255, 0.06);

  --semantic-success: #7fb069; /* iter KEPT */
  --semantic-danger: #ff6347; /* iter DISCARD */
  --semantic-warn: #ffd700; /* guard failure */
  --semantic-info: #9bc4e2;
}
```

**폰트 CDN 1줄**:

```html
<link
  href="https://fonts.googleapis.com/css2?family=Nanum+Pen+Script&family=Gowun+Dodum:wght@400;700&family=Caveat:wght@700&family=JetBrains+Mono:wght@500&display=swap"
  rel="stylesheet"
/>
```

| 용도                 | family                                  | 비고                                   |
| -------------------- | --------------------------------------- | -------------------------------------- |
| 페이지 헤드라인 (h1) | `'Nanum Pen Script', 'Caveat', cursive` | 회전 -1.2°, line-height 1.1            |
| 카드 제목            | `'Nanum Pen Script', cursive`           | 회전 -0.5°                             |
| 본문 (journal)       | `'Gowun Dodum', sans-serif`             | line-height 1.7, **손글씨체 금지**     |
| 캡션·메타            | `'Gowun Dodum', sans-serif`             | 12px, uppercase, letter-spacing 0.08em |
| 핵심 수치            | `'Caveat', 'Nanum Pen Script', cursive` | clamp(96px,12vw,192px), 옐로우         |
| 코드/diff            | `'JetBrains Mono', monospace`           | 배경 `--bg-chalkboard-deep`            |

**컴포넌트 매핑**:

| companion 영역    | Chalkboard 컴포넌트          | 핵심 규칙                                                   |
| ----------------- | ---------------------------- | ----------------------------------------------------------- |
| 헤더              | `SlideTitle` (intro)         | h1 회전 -1.2°, 우측 `ChalkBadge tomato` 로 진행률           |
| best metric 숫자  | `SlideStat` (yellow)         | numeric.hero 토큰. `%` 작게                                 |
| 메트릭 라인차트   | `ChalkSurface deep` + SVG    | line stroke `--stroke-chalk`, 데이터 점 강조 옐로우 1개만   |
| iter 카드 그리드  | `ChalkCard` (max 3 col)      | 카드 회전 각도 다르게 (-1.5°, +0.8°, -0.4°). 균일 정렬 금지 |
| iter verdict 배지 | `ChalkBadge`                 | KEEP=tomato, DISCARD=outline, CRASH=outline 점선            |
| 가설→결과 인용    | `SlideQuote` (yellow)        | 본문 강조 옐로우 단어 1개만                                 |
| git diff 미리보기 | `ChalkCallout` (info)        | 배경 `--bg-chalkboard-deep`, 코드 폰트                      |
| 가드 실패·crash   | `ChalkCallout` (warn)        | 옐로우 손그림 테두리 (2~3px wobble)                         |
| 섹션 구분         | `ChalkDivider` (dotted/wavy) | 균일하지 않게                                               |

**레이아웃 (적응판)**:

- 세로 스크롤, 섹션 폭 max 1200px 중앙 정렬.
- safe inset 좌우 desktop 96px / mobile 24px.
- iter 카드: desktop 3열 / tablet 2열 / mobile 1열. 그리드 룩 회피 (회전 각도·세로 위치 약간 어긋남).
- 옐로우 강조는 한 카드 안에 수치 또는 단어 1개만. 토마토+옐로우 동시 사용 금지.

**금지 사항** (companion.html 한정):

- 그라디언트, 글래스모피즘, `backdrop-filter`, blur 전환
- 컬러 드롭섀도 (분필 dust glow `rgba(255,255,255,0.06)`만 허용)
- 사진 배경
- corner radius 8px 초과
- 본문체에 손글씨체
- 강조에 밑줄 (밑줄 대신 색 변경)
- 컬러풀 이모지 (SVG 아이콘만)

**자가검증 5문항** (`companion.py` 출력 직전 codex가 점검):

1. 배경 `#2F4F4F` 단색? 그라디언트 없음?
2. 폰트는 손글씨 1 + 본문 1 + 모노 1 = 3종 이내?
3. 한 카드 안에 토마토+옐로우 동시 강조 없음?
4. iter 카드들의 회전 각도가 모두 동일하지 않음?
5. 큰 수치가 페이지에 1~2개? (3+ 면 분산)

## 8. STAGES/ 소크라테스 문안

각 단계 파일은 동일 4구역 구조: 목표 / 사전 점검 / 소크라테스 질문 / 통과 조건.

### 8.1 STAGES/00-setup.md — 환경 점검

- Q1. 진행 모드: ① codex 자동 보고 / ② 단계마다 수강생 직접 입력
- Q2. 작업 디렉토리 경로 확인 (기본 `~/workspace/autoresearch-class/`)

### 8.2 STAGES/01-skill-tour.md — autoresearch 스킬 이해

- Q1. Metric이 단일 숫자 stdout이어야 하는 이유는?
- Q2. "BatchNorm 추가 + LR 변경"을 한 번에 하면 어떤 학습 손실?
- Q3. Guard와 Metric의 차이를 한 문장으로.

### 8.3 STAGES/02-dataset.md — CIFAR-10

- Q1. `./data/` git 처리 방법: ① .gitignore ② Git LFS ③ 외부 캐시
- Q2. 다운로드 후 sanity check로 무엇을 확인? (클래스 분포·shape·정규화 통계)

### 8.4 STAGES/03-mlflow.md — MLflow 준비

- Q1. (a) `mlflow ui` 로컬 / (b) docker-compose self-host — docker 권장
- Q2. experiment vs run 차이. autoresearch 1회는 1 experiment에 매핑?

### 8.5 STAGES/04-baseline.md — 약한 베이스라인

- Q1. 베이스라인이 갖지 말아야 할 것은? (다중) BatchNorm / Augmentation / LR scheduler / Dropout / Adam
- Q2. baseline 1회 학습 후 val_acc는? (55~62% 예상) — 0번 iter 기준점.
- Q3. MLflow UI에서 params·metrics 확인 완료?

### 8.6 STAGES/05-journal.md — JOURNAL + Companion

- Q1. tsv가 있는데 JOURNAL.md를 별도로 두는 이유?
- Q2. JOURNAL 항목 1개의 필드? (iter # / 가설 / 변경 / 메트릭 / 해석 / 다음 / mlflow run_id)
- Q3. 빈 companion.html 미리 보고, 다음 단계 후 어떻게 바뀔지 예상.

### 8.7 STAGES/06-loop.md — `/autoresearch` 실행 ★

- Q1. Goal에 "75%" 임계치를 넣는 효과? (조기 종료)
- Q2. Scope에 data.py 포함 시 위험? (metric 정의 오염)
- Q3. Iterations 10 × 5분 = 50분.
- iter 중 인터랙션: 3연속 KEEP → "어떤 방향이 잘 듣고 있나?" / 2연속 DISCARD → "같은 함정에 빠지고 있나?"

### 8.8 STAGES/07-debrief.md — 회고

- Q1. 베이스라인 → 최종 best 개선 폭. 결정적이었던 iter?
- Q2. 시도하지 않은 개선 방향? (아키텍처 교체, pretrained, mixup)
- Q3. Guard를 더 엄격하게("val_acc 5%p 이내 하락") 두었다면?
- Q4. 본인 ML 프로젝트에 적용 시 Goal/Metric은?

## 9. 종료·가드 정책 (스킬 호출에 위임)

- 최대 iteration 상한: `Iterations: 10` (스킬 기본 기능).
- 목표 메트릭 달성 시 조기 종료: 스킬 기본 동작.
- 연속 K회 개선 실패 종료: 스킬 기본 동작 (rollback + discard 누적).
- 매 iter git 커밋 + 성능 하락 자동 rollback: 스킬 기본 동작 (`experiment:` prefix).

추가 가드는 두지 않는다 — 스킬이 이미 충분히 보호함.

## 10. 비목표 / 의도적 제외

- GPU 가속 강제: CPU만으로도 진행 가능한 학습 시간 유지.
- 멀티 데이터셋: CIFAR-10만.
- 외부 LLM API 직접 호출: codex CLI(OAuth)만 사용. API 키 노출 없음.
- 새로운 plugin/skill 패키지화: 본 repo는 *교육 자료*이지 *스킬 배포*가 아님.
- Streamlit/실시간 서버 대시보드: 정적 HTML로 충분.

## 11. 강사 전용 노트 (`docs/INSTRUCTOR.md`)

- 예상 실패 모드: 데이터 다운로드 네트워크 실패, MLflow 포트 충돌(5000), Python venv 누락.
- 토론 포인트: "Guard를 더 엄격하게 두면 안전하지만 탐색이 막힌다"의 trade-off, "Scope를 어디까지 열까"의 의사결정.
- 시연 시간 예산 (MPS 기준): 환경 점검 10분 + 베이스라인 10분 + 루프 50분 + 회고 10분 = 80분. CPU 환경이면 루프를 50~100분으로 늘리거나 `Iterations: 5`로 축소해 시연.

## 12. 향후 확장 (out of scope for v1)

- Fashion-MNIST·자체 한국어 이미지셋으로 데이터셋 교체 변형 실습.
- `/autoresearch:predict` 로 가설 사전 분석 추가.
- 졸업 연구 과제 템플릿 생성 (학생 본인 데이터·모델로 동일 흐름 적용).
