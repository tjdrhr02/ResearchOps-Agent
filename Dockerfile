# ── Stage 1: builder ─────────────────────────────────────────────
FROM python:3.11-slim AS builder
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# pyproject.toml 먼저 복사 → 의존성 레이어 독립 캐시
COPY pyproject.toml .

# build-system 없는 pyproject.toml 대응: tomllib(stdlib)로 의존성 추출 후 설치
RUN python3 -c "import tomllib,subprocess,sys; deps=tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']; subprocess.check_call([sys.executable,'-m','pip','install','--no-cache-dir']+deps)"

# ── Stage 2: runtime ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# 보안: 비루트 사용자
RUN useradd --system --create-home --uid 1000 appuser

WORKDIR /app

# 설치된 패키지와 선캐싱된 모델 복사
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
# 소스 코드 복사
COPY src/ ./src/

USER appuser

# Cloud Run은 $PORT 환경변수로 포트 지정 (기본값 8080)
EXPOSE 8080
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
