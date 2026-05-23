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
