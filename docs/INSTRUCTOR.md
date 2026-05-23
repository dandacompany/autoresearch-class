# Instructor Notes — autoresearch-class

수강생에게는 노출되지 않는 강사 전용 문서.

## 시간 예산 (MPS 기준)

| 단계          | 예상 시간                 |
| ------------- | ------------------------- |
| 00-setup      | 5분                       |
| 01-skill-tour | 10분                      |
| 02-dataset    | 5분                       |
| 03-mlflow     | 5분                       |
| 04-baseline   | 10분                      |
| 05-journal    | 5분                       |
| 06-loop       | 50분 (학습 5분 × 10 iter) |
| 07-debrief    | 10분                      |
| **합계**      | **약 100분**              |

CPU 환경이면 06-loop를 `Iterations: 5`로 축소하거나 시연을 사전 녹화로 대체.

## 예상 실패 모드

| 증상                                 | 원인                         | 대응                                     |
| ------------------------------------ | ---------------------------- | ---------------------------------------- |
| `mlflow` import 에러                 | venv 누락                    | `pip install -e .[dev]`                  |
| `localhost:5000` 접속 안 됨          | Docker 미실행 또는 포트 충돌 | `docker compose ps`, 충돌 시 5001로 변경 |
| `torchvision` CIFAR-10 다운로드 실패 | 네트워크 차단                | 사전 ZIP 배포해 `./data/` 에 압축 해제   |
| `/autoresearch` 가 같은 변경만 반복  | Scope 너무 좁음              | history 강조                             |

## 토론 포인트

- **Guard trade-off**: 더 엄격한 Guard는 안전하지만 탐색을 막는다. ML 안전·신뢰성 일반에 어떻게 매핑되는지.
- **Scope의 의미**: blast radius. data.py를 Scope에 포함하면 metric 정의가 오염될 수 있는 이유.
- **Mechanical metric 강제**: 왜 LLM judge가 아니라 stdout float인가. reproducibility와 reward hacking.

## 운영 팁

- 첫 강의는 반드시 본인 노트북에서 풀 리허설 1회.
- MLflow UI를 빔프로젝터에 띄워두고 진행 중 학생들이 볼 수 있게.
- 06-loop 시작 전 "10 iter 안에 75% 달성한다고 예측?" 사전 투표를 받으면 회고가 풍부해진다.
