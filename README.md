## FastAPI + LangGraph 챗봇 서버

이 프로젝트는 Spring Boot(WebSocket) 백엔드와 연동되는 FastAPI + LangGraph 기반 AI 챗봇 서버입니다.

- **상태 머신**: LangGraph로 `CHAT_0 ~ CHAT_6, CHAT_RESULT, RECOMMEND_RAG, BLUEPRINT_RAG` 등 단계 흐름 제어  
  - **수집 순서(UI 기준)**: `CHAT_0` 인트로 → `CHAT_1` 총예산(만원) → `CHAT_2` 평수 → `CHAT_3` 라이프스타일 → `CHAT_4` 보유+필요 가전(한 단계) → `CHAT_5` 구매계획 → `CHAT_6` 추천 생성 → 이후 **`RECOMMEND_RAG`**(추천 이후 일반 질의, 도면 없음) / **`BLUEPRINT_RAG`**(도면+패키지+스티커) / `CHAT_11` 종료  
  - **추천 12개·3개씩 보기**: `CHAT_6` 응답의 `all_recommendations` 전체를 **Spring이 보관**하고, 프론트는 **offset으로 3개씩**만 표시. **「다시 추천받기」** 는 Spring/프론트만 처리(FastAPI `CHAT_10` 없음). `RECOMMEND_RAG`에서는 “더 보여줘” 요청에 **새 패키지를 생성하지 않고** 안내 문구로 응답한다.  
  - **`CHAT_4`**: `userText`에 **`owned`(보유)** 와 **`needed` / `required`(필수·필요)** 를 **한 요청**에 보냄.
- **LLM**: OpenAI GPT-4o (langchain-openai)
- **데이터 모델**: Pydantic

**추천 알고리즘(EC2 `/data/recommendation_algorithm` 등)**  
- 템플릿(`pipeline.py`, `db.py` …)은 **수정하지 않습니다.**  
- `.env`에 `RECOMMENDATION_ALGORITHM_PATH`만 주고 **`RECOMMENDATION_ALGORITHM_ENTRYPOINT`를 비우거나 `pipeline:run_full_pipeline`** 이면, 내부 `pipeline_adapter`가 `run_full_pipeline(input_data, engine, use_llm=True)`를 호출합니다.  
  - `PATH`는 **`script/` 폴더의 부모**를 가리킵니다 (예: `…/recommendation_algorithm` — 그 안에 `script/pipeline.py`, `script/db.py` 가 있어야 함). 코드는 `script` 를 `sys.path`에 넣어 `from db import get_engine` 이 동작하게 합니다.  
- 파이프라인 **output** `packages`(예: 12세트)는 `recommendation_list`로 변환되며, 기존과 같이 **`display_recommendations`는 처음 3개**만 내려갑니다(프론트 “한 번에 3패키지”와 맞춤).  
- LangGraph `user_info` → 파이프라인 **input JSON** 매핑: 예산·평수·보유/필요 가전·라이프스타일(preferences)·style 등은 `pipeline_adapter.build_input_data_from_user_info`에서 구성(필드명은 팀 스펙에 맞게 조정 가능).  
- PATH만 있고 파이프라인이 아닌 다른 `모듈:함수`를 쓰려면 `RECOMMENDATION_ALGORITHM_ENTRYPOINT`를 그에 맞게 설정.  
- **미설정** 시 **LLM 폴백**으로 동작합니다.  
- 커스텀 진입 함수는 `(user_info, candidates)` 형태·반환은 `recommendation_list` dict 권장.

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

- `step`: 현재 단계 (`CHAT_1`, `CHAT_2`, …, `CHAT_RESULT`, `RECOMMEND_RAG`, `BLUEPRINT_RAG` 등)
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
- DynamoDB Local 기본 포트: `8001` → `AWS_ENDPOINT_URL=http://localhost:8001`

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

- `USER_INFO_CHAT`: 사용자 정보 파악(수집) 단계 (`CHAT_0~CHAT_5`)
- `PACKAGE_RECO_CHAT`: 패키지 추천/추천 이후 대화 (`CHAT_6` 요청 시 추천 생성 후 내부 `CHAT_RESULT`, 이후 `RECOMMEND_RAG` / `BLUEPRINT_RAG` 등)
- **`RECOMMEND_RAG`**: 추천 이후 **일반 질의**(가격·스펙·비교, “다음 추천 보여줘” 등). **도면·배치 좌표는 다루지 않음** — `userText`에 `floorPlanId`/`placements`가 있어도 **무시**됩니다.
- **`BLUEPRINT_RAG`**: **도면 단계** — 도면 옵션 + 선택 패키지 인덱스 + 스티커 좌표를 `userText`로 보내면, 같은 `convId`의 `all_recommendations`와 DB 제품 상세로 **배치 적합성 RAG** (`aiResponse` + `data.blueprint_rag`).

#### `BLUEPRINT_RAG` 요청 예시 (도면 + 패키지 + 스티커)

**전제**: 동일 `convId`로 `CHAT_6`까지 완료되어 체크포인트에 `all_recommendations`가 있어야 합니다.

```json
{
  "convId": "chat_T6bC9dE2fG5hJ",
  "stepCode": "BLUEPRINT_RAG",
  "userText": {
    "floorPlanId": "fp_opt_2",
    "packageIndex": 0,
    "floorPlanImageUrl": "https://example.com/floorplans/fp_opt_2.png",
    "canvasSize": { "width": 800, "height": 600 },
    "placements": [
      { "itemType": "refrigerator", "x": 0.12, "y": 0.45 },
      { "itemType": "washing_machine", "x": 0.72, "y": 0.38 }
    ],
    "utilities": [
      { "type": "drainage", "x": 0.7, "y": 0.4 },
      { "type": "outlet", "x": 0.15, "y": 0.5 }
    ]
  }
}
```

- `packageIndex`: `all_recommendations` 배열의 **0부터** 인덱스.
- `placements` / `utilities`의 `x`,`y`는 캔버스 대비 **정규화(0~1)** 또는 픽셀 — 프론트 규약에 맞게 보내면 됩니다.
- 응답: `aiResponse`에 요약 답변, `data.blueprint_rag.llm`에 `answer`, `placement_warnings`, `suggestions` 등 JSON.

#### `RECOMMEND_RAG` (도면 없음 — 옆 채팅 등)

- **질문 + (선택) 패키지 인덱스 + (선택) 상품명 매칭**만 사용합니다. **세탁기·배수 위치처럼 도면이 필요한 질문은 `BLUEPRINT_RAG`로 보내세요.**

```json
{
  "convId": "chat_T6bC9dE2fG5hJ",
  "stepCode": "RECOMMEND_RAG",
  "userText": {
    "message": "1번 패키지 냉장고 용량이 얼마야?",
    "packageIndex": 0
  }
}
```

또는 문자열만:

```json
{
  "convId": "chat_T6bC9dE2fG5hJ",
  "stepCode": "RECOMMEND_RAG",
  "userText": "LG 디오스 냉장고 깊이 알려줘"
}
```

- `packageIndex`는 Spring에 저장한 값을 붙이면 됩니다. 생략 시 질문의 `1번` 등 또는 이전 `BLUEPRINT_RAG`에서 저장된 `user_info.selected_package_index`를 사용할 수 있습니다.
- **상품명만 입력**: `packageIndex` 없이도 `all_recommendations`의 `product_name`·`model_id`와 질문을 매칭해 DB 스펙을 붙일 수 있습니다(휴리스틱).

##### 채팅으로 「다음 추천 / 더 보여줘」 — Spring 연동

`RECOMMEND_RAG` 응답 `data`에 다음이 포함될 수 있습니다.

- `show_next_recommendation_page` (또는 동일 값 `showNextRecommendationPage`): **`true`** 이면 사용자가 **이미 내려준 전체 목록(예: 12개)에서 다음 3개**를 보여 달라는 의미로 처리한 것입니다. (LLM + 키워드 보조)

**Spring에서 할 일 (예시):**

1. `CHAT_6` 때 받은 `all_recommendations` 전체를 세션/DB에 저장.
2. 화면에는 `offset` 기준으로 `list.subList(offset, min(offset+3, size))` 만 표시.
3. `RECOMMEND_RAG` 응답 후 `data.show_next_recommendation_page == true` 이면 `offset += 3` (끝이면 0으로 돌리거나 버튼 비활성).
4. 채팅창에는 `aiResponse`(예: 네, 옆에서 다음 추천을 확인해 주세요)만 표시하면 됨.

`false` 이면 일반 질의응답이므로 offset은 건드리지 않습니다.

