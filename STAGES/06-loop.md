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
