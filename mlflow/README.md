# MLflow server (self-hosted)

```bash
cd mlflow
docker compose up -d
open http://localhost:5000
docker compose down            # stop (preserves data)
docker compose down -v         # reset (drops postgres volume)
```

학습 스크립트는 `MLFLOW_TRACKING_URI=http://127.0.0.1:5000`을 기본값으로 사용합니다.
