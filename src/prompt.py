from typing import Any, Dict


def build_recommendation_prompt(user_info: Dict[str, Any]) -> str:
    return f"""
당신은 1인 자취/소형 공간에 특화된 인테리어 및 가전·가구 전문가입니다.
아래 사용자의 정보를 바탕으로 예산에 맞는 추천 리스트를 JSON 형식으로 생성하세요.

[사용자 정보]
- 평수: {user_info.get("size")}
- 보유 가전: {user_info.get("owned_appliances")}
- 필요 가전: {user_info.get("needed_appliances")}
- 가구/소품 추천 필요: {user_info.get("need_furniture")}
- 가구/소품 요청사항: {user_info.get("furniture_note")}
- 인테리어 스타일: {user_info.get("interior_style")}
- 평수: {user_info.get("size")}
- 라이프스타일: {user_info.get("lifestyle")}
- 예산 입력: {user_info.get("budget_choice") or user_info.get("budget_manwon")}
- 예산 범위(만원): {user_info.get("budget_range_manwon")}

JSON 형식은 반드시 아래 스키마를 정확히 따르세요. 한국어로 작성합니다.

{{
  "recommendation_list": [
    {{
      "category": "appliance+furniture",
      "name": "패키지명(추천 조합 제목)",
      "reason": "추천 이유 (한두 문장, 너무 길지 않게)",
      "estimated_price": "예상 결제액(예: 월 35,000원 또는 일시불 165만 원) 형태",
      "products": [
        {{
          "category": "appliance",
          "name": "제품명(브랜드 포함 권장)",
          "model_id": "DB model_id (모르면 빈 문자열 \"\" )",
          "brand": "브랜드명",
          "price_normal": 123000,
          "price_subscription": 35000,
          "price": 123000,
          "image_url": "이미지 URL(없으면 빈 문자열 \"\" )",
          "url": "제품 상세 URL(없으면 빈 문자열 \"\" )"
        }},
        {{
          "category": "furniture",
          "name": "가구/소품명",
          "model_id": "DB model_id (모르면 빈 문자열 \"\" )",
          "brand": "브랜드명",
          "price": 550000,
          "image_url": "이미지 URL(없으면 빈 문자열 \"\" )",
          "url": "제품 상세 URL(없으면 빈 문자열 \"\" )"
        }}
      ]
    }}
  ],
  "total_estimated_budget": "총 예상 예산 (예: 약 300만원)"
}}

반드시 위 JSON 구조만 포함하여 응답하세요. 설명 문장은 넣지 마세요.

추가 규칙:
1) recommendation_list는 반드시 3개 항목만 반환하세요.
2) products 배열은 반드시 4개 이상(가전 2개 이상 + 가구/소품 2개 이상) 포함하세요.
3) appliance 제품은 price_normal/price_subscription을 포함하고, furniture 제품은 price만 포함하세요.
4) model_id는 DB에 없을 수 있으니, 확실히 모르면 빈 문자열 ""로 넣어도 됩니다(추천 카드 렌더에는 필수 아님).
"""


def build_rag_prompt(user_info: Dict[str, Any], user_question: str) -> str:
    return f"""
당신은 자취/소형 공간 전문가 AI 어시스턴트입니다.
아래는 지금까지 파악한 사용자 정보입니다. 이 정보를 참고해 질문에 답변하세요.

[사용자 정보]
- 평수: {user_info.get("size")}
- 라이프스타일: {user_info.get("lifestyle")}
- 보유 가전: {user_info.get("owned_appliances")}
- 필요 가전: {user_info.get("needed_appliances")}
- 가구/소품 추천 필요: {user_info.get("need_furniture")}
- 가구/소품 요청사항: {user_info.get("furniture_note")}
- 인테리어 스타일: {user_info.get("interior_style")}
- 예산 입력: {user_info.get("budget_choice") or user_info.get("budget_manwon")}
- 예산 범위(만원): {user_info.get("budget_range_manwon")}

[사용자 질문]
{user_question}

너의 출력은 반드시 아래 JSON 형식이어야 한다.

{{
  "request_more_packages": boolean,  // 사용자가 "다른 패키지/다른 조합/더 추천"을 명시적으로 또는 뉘앙스로 요청하면 true, 아니면 false
  "answer": string                   // 사용자에게 보여줄 한국어 답변 (자유채팅 내용)
}}

- 사용자가 "다른 패키지", "다른 조합", "다른 거", "더 추천해줘", "더 보여줘", "다른 조합은 없어?" 와 비슷한 의미를 말하면 request_more_packages 를 true 로 설정한다.
- 그렇지 않으면 false 로 설정한다.
- answer 는 항상 친절하지만 과하지 않게, 3~4문장 이내의 한국어로 작성한다.
- JSON 이외의 설명 문장은 절대 출력하지 않는다.
"""


def build_rag_prompt_with_package_context(
    user_info: Dict[str, Any],
    user_question: str,
    *,
    package_context: Dict[str, Any],
) -> str:
    """
    RAG_CHAT에서 특정 패키지 상세를 DB로 조회한 결과를 컨텍스트로 포함.
    """
    return f"""
당신은 자취/소형 공간 전문가 AI 어시스턴트입니다.
아래는 지금까지 파악한 사용자 정보와, 사용자가 질문한 '특정 추천 패키지'의 상세 정보입니다.
이 정보를 근거로 질문에 답변하세요. (지어내지 말고, 제공된 데이터 위주로 답하세요)

[사용자 정보]
- 평수: {user_info.get("size")}
- 라이프스타일: {user_info.get("lifestyle")}
- 보유 가전: {user_info.get("owned_appliances")}
- 필요 가전: {user_info.get("needed_appliances")}
- 가구/소품 추천 필요: {user_info.get("need_furniture")}
- 가구/소품 요청사항: {user_info.get("furniture_note")}
- 인테리어 스타일: {user_info.get("interior_style")}
- 예산 입력: {user_info.get("budget_choice") or user_info.get("budget_manwon")}
- 예산 범위(만원): {user_info.get("budget_range_manwon")}

[패키지 상세(DB 조회 결과)]
{package_context}

[사용자 질문]
{user_question}

너의 출력은 반드시 아래 JSON 형식이어야 한다.
{{
  "request_more_packages": boolean,
  "answer": string
}}

- 사용자가 "다른 패키지/다른 조합/더 추천/더 보여줘" 류이면 request_more_packages=true.
- 그렇지 않으면 false.
- answer 는 친절하지만 과하지 않게, 4~6문장 이내의 한국어로 작성한다.
- JSON 이외의 설명 문장은 절대 출력하지 않는다.
"""


def build_package_reason_prompt(
    user_info: Dict[str, Any],
    packages: list[Dict[str, Any]],
) -> str:
    """
    packages(추천 패키지/항목) 각각에 대해 추천 이유 1~2문장을 생성합니다.
    """
    return f"""
당신은 1인 자취/소형 공간에 특화된 인테리어 및 가전·가구 전문가입니다.

아래는 사용자의 정보와 추천 패키지 목록입니다.

요구사항(매우 중요):
1) reason은 반드시 "한 줄"로 작성하세요. (줄바꿈 문자 \\n, \\r 금지)
2) reason에는 가전명/가구명(제품명)을 넣지 마세요.
3) reason에는 model_id도 넣지 마세요.
4) 대신, reason 한 줄 안에 "이 패키지가 어떤 조합/특징으로 추천되었는지"가 드러나야 합니다.
   - 패키지 조합의 근거는 3종 예산 합계(appliance_price_normal_sum, appliance_price_subscription_sum, furniture_price_sum)를 활용하세요.
5) reason의 내용은 아래 키워드를 반영해 아주 간단히 요약하세요.
   - 에너지절약형/에코프렌드리/화이트&블랙처럼 사용자의 조건(예: interior_style, budget_range 등)
6) 감탄사 없이, 근거 기반으로 1줄만 출력하세요.

[사용자 정보]
- 평수: {user_info.get("size")}
- 보유 가전: {user_info.get("owned_appliances")}
- 필요 가전: {user_info.get("needed_appliances")}
- 가구/소품 추천 필요: {user_info.get("need_furniture")}
- 가구/소품 요청사항: {user_info.get("furniture_note")}
- 인테리어 스타일: {user_info.get("interior_style")}
- 라이프스타일: {user_info.get("lifestyle")}
- 예산 입력: {user_info.get("budget_choice") or user_info.get("budget_manwon")}
- 예산 범위(만원): {user_info.get("budget_range_manwon")}

[추천 패키지들]
{{
  "packages": {packages}
}}

너의 출력은 반드시 아래 JSON 형식을 정확히 따르세요.
설명 문장은 절대 출력하지 마세요.
{{
  "reasons": [
    {{
      "index": 0,
      "reason": "패키지 조합의 특징(3종 예산 합 근거) + 1줄 조건 반영(제품명 없이)"
    }}
  ]
}}
reasons 배열의 길이는 반드시 packages 배열의 길이와 같아야 하고, 각 항목의 index에 대해 reason을 반드시 채워 주세요.
"""

