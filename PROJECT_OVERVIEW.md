# ResearchOps Agent

## 프로젝트 개요

ResearchOps Agent는 논문, 기술 블로그, 뉴스 정보를 자동으로 수집하고 분석하여 **Research Brief**를 생성하는 Agentic AI 서비스이다.

이 프로젝트의 목적은 단순한 챗봇을 만드는 것이 아니라 다음을 경험하는 것이다.

- Agentic AI workflow 설계
- LangChain 기반 AI 서비스 구현
- Tool 기반 외부 데이터 수집 (arXiv, DuckDuckGo)
- RAG 기반 근거 검색 시스템 (로컬 임베딩)
- FastAPI 기반 비동기 AI 서비스
- Observability 기반 AI 운영 지표

이 프로젝트는 **AI 시스템을 설계할 줄 아는 엔지니어** 역량을 보여주는 것을 목표로 한다.

---

# 문제 정의

AI / 기술 분야의 정보는 다음과 같은 특징이 있다.

- 논문
- 기술 블로그
- 뉴스
- 컨퍼런스 발표

이 모든 정보가 **매우 빠르게 증가**하고 있다.

개발자가 특정 주제를 조사하려면 다음 작업이 반복된다.

1. 논문 검색
2. 기술 블로그 검색
3. 뉴스 검색
4. 핵심 내용 요약
5. 서로 다른 관점 비교
6. 정리된 연구 브리프 작성

이 작업은 매우 많은 시간을 요구한다.

따라서 이 프로젝트는 다음 문제를 해결한다.

> 연구 질문을 입력하면 관련 정보를 수집하고 분석하여 Research Brief를 자동 생성하는 AI Agent

---

# 핵심 기능

ResearchOps Agent는 다음 기능을 제공한다.

### 1 Research Workflow 자동화

사용자 질문 → 연구 계획 → 정보 수집 → RAG 인덱싱/검색 → 분석 → 연구 브리프 생성

### 2 Multi-Agent 시스템

다음 Agent로 구성된다.

- **PlannerAgent** — 쿼리 분석, 검색 전략 및 source_priority 결정
- **CollectorAgent** — Tool 기반 실제 데이터 수집
- **SynthesizerAgent** — 수집 데이터 + RAG 증거 분석, 관점 비교
- **ReporterAgent** — 최종 Research Brief 생성

### 3 Tool 기반 실제 데이터 수집

다음 데이터를 자동 수집한다.

- **논문** — arXiv REST API (무료, 키 불필요, 최신 논문 Atom XML)
- **기술 블로그** — DuckDuckGo 텍스트 검색 (ddgs 패키지, Medium/dev.to/TDS 등)
- **뉴스** — DuckDuckGo 뉴스 검색 (ddgs 패키지)
- **아티클 본문** — httpx + BeautifulSoup4 크롤링

### 4 RAG 기반 근거 검색

수집된 문서를 벡터화하여 의미론적 검색을 수행한다.

- **임베딩** — sentence-transformers `all-MiniLM-L6-v2` (384차원, 로컬 실행)
- **청킹** — SemanticChunker (500자, 80자 오버랩)
- **벡터 저장소** — InMemoryVectorStore (코사인 유사도 검색)
- **검색** — retrieve_multi_query (복수 쿼리 기반 증거 추출)

### 5 Research Brief 생성

최종 결과물은 다음 형태이다.

- **executive_summary** — 의사결정자용 3~4문장 요약
- **key_trends** — 주요 트렌드 (출처 타입 포함)
- **evidence** — RAG 검색된 핵심 증거 (URL, 유사도 점수 포함)
- **source_comparison** — 소스 유형별 관점 비교
- **citations** — 실제 URL 기반 인용 목록
- **open_questions** — 후속 연구 방향

### 6 비동기 잡 처리

- `POST /research/run` → 즉시 `job_id` 반환 (202 Accepted)
- `GET /research/{job_id}` → polling으로 완료 여부 확인
- `GET /research/{job_id}/sources` → 수집된 소스 목록 조회

---

# 기술 스택

### Backend
- Python 3.11+
- FastAPI (비동기 HTTP, Depends DI)
- uvicorn

### AI Framework
- LangChain (PydanticOutputParser, ChatGoogleGenerativeAI)
- Gemini 2.5 Flash (Planner / Synthesizer / Reporter LLM)
- sentence-transformers `all-MiniLM-L6-v2` (로컬 임베딩)

### 데이터 수집
- arXiv REST API (논문)
- ddgs / DuckDuckGo Search (블로그, 뉴스)
- httpx + BeautifulSoup4 (아티클 크롤링)

### 저장소 (현재)
- InMemoryVectorStore (벡터 검색, 휘발성)
- InMemoryRedisClient (노트 저장, 휘발성)

### 저장소 (프로덕션 교체 대상)
- PostgreSQL + pgvector (영속성 벡터 저장소)
- Redis (분산 캐시)

### Observability
- MetricsCollector (요청 수, 실패 수, latency 추적)
- 구조적 로깅 (단계별 elapsed_ms, step_trace)

---

# 에러 격리 전략

각 단계의 실패가 전체 워크플로우를 중단시키지 않도록 격리되어 있다.

| 단계 | 실패 시 동작 |
|------|------------|
| PlannerAgent | **FATAL** — plan 없이 진행 불가, 워크플로우 중단 |
| CollectorAgent | 빈 문서 목록으로 계속 |
| RAG (index/retrieve) | 빈 evidence로 계속 |
| SynthesizerAgent | 빈 synthesis로 계속 |
| ReporterAgent | **FATAL** — brief 없이 반환 불가, 워크플로우 중단 |

---

# 향후 확장 포인트

| 항목 | 현재 | 프로덕션 |
|------|------|---------|
| 임베딩 | 로컬 all-MiniLM-L6-v2 (384차원) | OpenAI/Google 고품질 임베딩 |
| 벡터 저장소 | InMemoryVectorStore (휘발성) | pgvector (영속성) |
| 잡 큐 | asyncio.create_task (단일 프로세스) | Celery + Redis (분산) |
| 검색 소스 | arXiv, DuckDuckGo (비공식) | Semantic Scholar, NewsAPI (공식) |
| 워크플로우 | 선형 파이프라인 | LangGraph 비선형 워크플로우 |

---

# 프로젝트 목표

이 프로젝트는 단순 기능 구현이 아니라 다음을 목표로 한다.

- AI 시스템 설계 능력
- Agent architecture 이해
- RAG 시스템 구현 경험
- AI 서비스 운영 관점 이해
