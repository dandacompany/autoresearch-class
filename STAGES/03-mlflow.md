## 목표

self-hosted MLflow 서버를 docker-compose로 띄우고 UI 접속을 확인한다.

## 사전 점검

```bash
cd mlflow && docker compose up -d
sleep 8
curl -fsS http://127.0.0.1:5000/health
```

`OK` 가 출력되면 정상.

## 소크라테스 질문

1. "MLflow를 (a) `mlflow ui` 로컬 단일 프로세스 vs (b) docker-compose self-host 중 어느 쪽으로 띄울까요? 강의에서는 (b)를 권장하는데 그 이유는?"
   - 힌트: 종료 후 재기동 시 run 기록이 보존되는가
2. "MLflow의 'experiment'와 'run'의 차이는 무엇일까요? autoresearch 1회 호출은 1 experiment에 매핑되는 것이 자연스러울까요, 아니면 1 run인가요?"
   - 정답 컨벤션: 1 experiment(autoresearch-class), N runs (iter별 1 run)
3. "수강생이 매 iter 후 MLflow UI에서 무엇을 보면 좋을까요?" (Compare 기능, val_acc 라인 차트, params diff)

## 통과 조건

- [ ] `http://localhost:5000` UI 로드됨
- [ ] `autoresearch-class` experiment가 자동 생성될 준비됨
- [ ] 수강생이 experiment vs run 차이를 설명함
