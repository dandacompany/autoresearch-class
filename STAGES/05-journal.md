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
