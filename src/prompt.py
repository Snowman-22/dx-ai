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
      "category": "가전 또는 가구",
      "name": "제품명",
      "reason": "추천 이유 (한두 문장)",
      "estimated_price": "예상 가격 범위 (예: 50~70만원)"
    }}
  ],
  "total_estimated_budget": "총 예상 예산 (예: 약 300만원)"
}}

반드시 위 JSON 구조만 포함하여 응답하세요. 설명 문장은 넣지 마세요.
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

