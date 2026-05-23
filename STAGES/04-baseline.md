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
