# ResearchOps Agent

## 프로젝트 개요

ResearchOps Agent는 논문, 기술 블로그, 뉴스 정보를 자동으로 수집하고 분석하여 **Research Brief**를 생성하는 Agentic AI 서비스이다.

이 프로젝트의 목적은 단순한 챗봇을 만드는 것이 아니라 다음을 경험하는 것이다.

- Agentic AI workflow 설계
- LangChain 기반 AI 서비스 구현
- Tool 기반 외부 데이터 수집
- RAG 기반 근거 검색 시스템
- FastAPI 기반 AI 서비스
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

사용자 질문 → 연구 계획 → 정보 수집 → 분석 → 연구 브리프 생성

### 2 Multi-Agent 시스템

다음 Agent로 구성된다.

- Planner Agent
- Collector Agent
- Synthesizer Agent
- Reporter Agent

### 3 Tool 기반 데이터 수집

다음 데이터를 자동 수집한다.

- 논문
- 기술 블로그
- 뉴스

### 4 RAG 기반 근거 검색

수집된 문서 및 저장된 노트를 기반으로 검색한다.

### 5 Research Brief 생성

최종 결과물은 다음 형태이다.

- 핵심 요약
- 주요 트렌드
- 핵심 출처
- 관점 비교
- 향후 연구 방향

---

# 기술 스택

### Backend
Python  
FastAPI  

### AI Framework
LangChain  

### Database
PostgreSQL + pgvector  

### Cache
Redis  

### Observability
metrics tracking  
token usage tracking  
latency tracking  

---

# 최종 결과물

이 프로젝트는 다음 결과물을 생성한다.

- AI Research Agent 서비스
- Agentic workflow 구현
- Tool 기반 데이터 수집
- RAG 기반 문서 검색
- Research Brief 생성
- AI Observability 지표

---

# 프로젝트 목표

이 프로젝트는 단순 기능 구현이 아니라 다음을 목표로 한다.

- AI 시스템 설계 능력
- Agent architecture 이해
- RAG 시스템 구현 경험
- AI 서비스 운영 관점 이해