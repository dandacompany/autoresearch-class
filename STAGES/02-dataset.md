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
