## FastAPI + LangGraph 챗봇 서버

이 프로젝트는 Spring Boot(WebSocket) 백엔드와 연동되는 FastAPI + LangGraph 기반 AI 챗봇 서버입니다.

- **상태 머신**: LangGraph로 `CHAT_0 ~ CHAT_RESULT, RAG_CHAT` 단계 흐름 제어
- **LLM**: OpenAI GPT-4o (langchain-openai)
- **데이터 모델**: Pydantic

**저장소 역할**  
- **Spring Boot + PostgreSQL**: `Chat` 테이블 등으로 구조화된 메타데이터(chat_id, guest_session_id, start_date 등) 관리.  
- **DynamoDB**: 한 대화(채팅방)의 모든 채팅 내용·상태를 **`conv_id`**(LangGraph `thread_id`) 기준으로 저장해 같은 대화 흐름을 이어갑니다.

### 1. 환경 변수

`.env` 파일(직접 생성)에 아래 값을 설정하세요.

```bash
OPENAI_API_KEY=sk-...
HOST=0.0.0.0
PORT=8000
```

#### DynamoDB 연결 (선택)

세션 상태를 DynamoDB에 저장하려면 다음을 추가하세요.

```bash
# 테이블 이름 지정 시 DynamoDB 체크포인터 사용 (미설정 시 메모리 사용)
DYNAMODB_TABLE_NAME=langgraph-chat-checkpoints
# TTL(초, 선택) - 항목 자동 삭제
DYNAMODB_TTL_SECONDS=86400
```

AWS 인증은 아래 중 하나로 설정합니다.

- **환경 변수**: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- **프로파일**: `~/.aws/credentials`, `AWS_PROFILE`
- **IAM 역할**: EC2/ECS/Lambda 등에서 역할 사용 시 별도 설정 불필요

리전은 `AWS_REGION` 또는 `AWS_DEFAULT_REGION`(예: `ap-northeast-2`)으로 지정합니다.  
테이블이 없으면 `langgraph-dynamodb-checkpoint`가 자동 생성합니다(PK: thread_id, SK: checkpoint_id).

#### DynamoDB에 저장되는 형태

체크포인터가 쓰는 테이블 구조는 다음과 같습니다.

| 항목 | 설명 |
|------|------|
| **Partition Key (PK)** | `thread_id` → 우리는 `conv_id`를 그대로 사용 (예: `chat_T6bC9dE2fG5hJ`) |
| **Sort Key (SK)** | `checkpoint_id` → 요청/노드 실행 시점마다 생성되는 고유 ID (예: UUID) |
| **저장 내용** | 해당 시점의 **그래프 상태 스냅샷** (LangGraph가 직렬화한 체크포인트) |

체크포인트 본문에는 아래와 같은 **채널 값**이 들어갑니다 (우리 `ChatState`에 대응).

- `step`: 현재 단계 (`CHAT_1`, `CHAT_2`, …, `CHAT_RESULT`, `RAG_CHAT`)
- `user_info`: 누적된 사용자 정보 (예산, 평수, 라이프스타일, 보유/필요 가전 등)
- `messages`: 대화 메시지 목록 (현재는 미사용)
- 그 외 LangGraph가 사용하는 메타데이터(버전, 체크포인트 네임스페이스 등)

즉, **같은 `conv_id`(thread_id)로 여러 번 요청**이 오면, 이전에 저장된 `user_info`·`step`을 복구해서 다음 단계를 이어가고, 실행 후 새 스냅샷을 다시 DynamoDB에 적재합니다.  
대화 문구(`question_text`·`user_text`) 이력 저장은 Spring Boot가 MongoDB 등에 하는 것이고, FastAPI/DynamoDB에는 **세션 상태(체크포인트)** 만 저장됩니다.

**로컬 DynamoDB** (Localstack, DynamoDB Local) 사용 시:

```bash
DYNAMODB_TABLE_NAME=langgraph-chat-checkpoints
AWS_ENDPOINT_URL=http://localhost:4566
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=test
AWS_SECRET_ACCESS_KEY=test
```

- Localstack 기본 포트: `4566`
- DynamoDB Local 기본 포트: `8000` → `AWS_ENDPOINT_URL=http://localhost:8000`

**로컬 → 실제 AWS 전환**: `.env`에서 `AWS_ENDPOINT_URL`을 제거(또는 주석 처리)하고, `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`를 실제 키로 바꾸면 됩니다. 코드 수정 없이 환경 변수만으로 전환 가능합니다.

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2-1. (선택) Postgres pgvector 설정

`products` 테이블에 벡터 컬럼(`embedding vector(1536)`)과 인덱스를 추가하려면 아래 SQL을 실행하세요.

- 파일: `sql/pgvector_setup.sql`

예시(로컬):

```bash
psql -h localhost -p 5432 -U postgres -d snowman -f sql/pgvector_setup.sql
```

### 3. 서버 실행

```bash
python main.py
```

또는

```bash
uvicorn app:app --reload
```

### 4. Spring Boot ↔ FastAPI 연동 규약

- **엔드포인트**: `POST /ai/chat`
- **요청 바디**:

```json
{
  "conv_id": "chat_T6bC9dE2fG5hJ",
  "step_code": "CHAT_1",
  "assistant_text": "프론트/스프링이 화면에 표시한 assistant 문구(선택)",
  "user_text": "사용자 입력 (string | string[] | object)"
}
```

- **응답 바디**:

```json
{
  "data": {},
  "ai_response": null
}
```

`step_code`는 **프론트 → Spring Boot → FastAPI**로 전달되는 "현재 단계"입니다.  
FastAPI는 해당 단계에 맞는 노드를 실행하고, 세션 상태는 `conv_id`(LangGraph `thread_id`) 기준으로 체크포인트에 저장/복구됩니다.

`chat_type`은 채팅 유형 구분용입니다.

- `USER_INFO_CHAT`: 사용자 정보 파악(수집) 단계 (`CHAT_0~CHAT_6`, `CHAT_3_1`)
- `PACKAGE_RECO_CHAT`: 패키지 추천/추천 이후 대화 (`CHAT_RESULT`, `RAG_CHAT`)

