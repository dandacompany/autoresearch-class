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
