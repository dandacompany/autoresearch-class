## 목표

수강생의 환경(codex CLI OAuth, `/autoresearch` 스킬, Python 3.11+, Docker)이 갖춰졌는지 확인하고, 작업 디렉토리를 합의한다.

## 사전 점검

```bash
codex --version
which docker && docker version --format '{{.Server.Version}}'
python3 --version
ls -d ~/.claude/plugins/cache/autoresearch 2>/dev/null && echo "autoresearch skill present" || echo "autoresearch skill MISSING"
```

위 명령을 codex가 실행하고 결과를 보고합니다.

## 소크라테스 질문

1. (자동 점검 결과를 보여준 뒤) "진행 모드를 어떻게 할까요?"
   - ① 모든 단계를 codex가 자동 보고 → 수강생은 OK만
   - ② 단계마다 수강생이 직접 명령 입력 → codex는 안내만
2. "작업 디렉토리는 어디로 할까요?" (기본 `~/workspace/autoresearch-class/`)
3. "Docker가 띄워져 있나요? 없으면 Docker Desktop을 켜고 다시 알려주세요."

## 통과 조건

- [ ] codex CLI 인증됨
- [ ] `/autoresearch` 스킬 설치 확인됨
- [ ] Python 3.11+ 확인됨
- [ ] Docker 데몬 동작 중
- [ ] 작업 디렉토리 합의됨
