import json
from typing import Any, Dict, List, Optional, Union


def build_recommendation_prompt(
    user_info: Dict[str, Any],
    *,
    candidate_products: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    [사용 중단] 과거에는 node_chat_result 가 NotImplementedError 시 LLM으로 추천 JSON을 만들 때 썼음.
    현재 graph 는 이 프롬프트를 호출하지 않음. LLM은 스키마 예시만 보고도 example.com 같은 가짜 URL·상품을 지어낼 수 있음.
    """
    catalog_json = json.dumps(candidate_products or [], ensure_ascii=False)
    return f"""
당신은 1인 자취/소형 공간에 특화된 인테리어 및 가전·가구 전문가입니다.
아래 사용자의 정보를 바탕으로 예산에 맞는 추천 리스트를 JSON 형식으로 생성하세요.

[사용자 정보]
- 평수: {user_info.get("size")}
- 라이프스타일: {user_info.get("lifestyle")}
- 보유 가전: {user_info.get("owned_appliances")}
- 필요 가전: {user_info.get("needed_appliances")}
- 새 공간 구매 계획(가구/가전): {user_info.get("purchase_plans")}
- 가구/소품 추천 필요: {user_info.get("need_furniture")}
- 가구/소품 요청사항(레거시): {user_info.get("furniture_note")}
- 인테리어 스타일(레거시): {user_info.get("interior_style")}
- 예산 입력: {user_info.get("budget_choice") or user_info.get("budget_manwon")}
- 예산 범위(만원): {user_info.get("budget_range_manwon")}

[상품 카탈로그(product 테이블에서 조회됨)]
{catalog_json}

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
          "product_name": "제품명(브랜드 포함 권장)",
          "model_id": "DB model_id (모르면 빈 문자열 \"\" )",
          "brand": "브랜드명",
          "price_normal": 123000,
          "price_subscription": 35000,
          "price": 123000,
          "product_image_url": "이미지 URL(없으면 빈 문자열 \"\" )",
          "product_url": "제품 상세 URL(없으면 빈 문자열 \"\" )"
        }},
        {{
          "category": "furniture",
          "product_name": "가구/소품명",
          "model_id": "DB model_id (모르면 빈 문자열 \"\" )",
          "brand": "브랜드명",
          "price": 550000,
          "product_image_url": "이미지 URL(없으면 빈 문자열 \"\" )",
          "product_url": "제품 상세 URL(없으면 빈 문자열 \"\" )"
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
5) 반드시 위 [상품 카탈로그] 안에 있는 상품만 사용하세요. 임의 상품명 생성 금지.
6) 각 product의 model_id, product_url, product_image_url은 카탈로그 값을 그대로 사용하세요.
"""


def build_rag_prompt(
    user_info: Dict[str, Any],
    user_question: str,
) -> str:
    return f"""
당신은 자취/소형 공간 전문가 AI 어시스턴트입니다.
아래는 지금까지 파악한 사용자 정보입니다. 이 정보를 참고해 질문에 답변하세요.
(도면·배치 좌표는 이 단계에서 다루지 않습니다. 일반 질의·추천 패키지/상품 스펙 질문에만 답합니다.)

[사용자 정보]
- 평수: {user_info.get("size")}
- 라이프스타일: {user_info.get("lifestyle")}
- 보유 가전: {user_info.get("owned_appliances")}
- 필요 가전: {user_info.get("needed_appliances")}
- 새 공간 구매 계획(가구/가전): {user_info.get("purchase_plans")}
- 가구/소품 추천 필요: {user_info.get("need_furniture")}
- 가구/소품 요청사항(레거시): {user_info.get("furniture_note")}
- 인테리어 스타일(레거시): {user_info.get("interior_style")}
- 예산 입력: {user_info.get("budget_choice") or user_info.get("budget_manwon")}
- 예산 범위(만원): {user_info.get("budget_range_manwon")}

[사용자 질문]
{user_question}

추가 규칙(매우 중요):
- 사용자가 제품의 '크기/치수/가로/세로/높이/깊이/폭/두께'처럼 구체적인 실측을 물으면, 제공된 DB 기반 '제품 스펙(width/height/depth)'이 있는 경우 그 숫자를 그대로 사용해 문장을 구성하세요.
- 스펙 숫자가 제공되지 않았다면 추정/대략값을 만들지 말고 "정확한 스펙은 확인이 필요합니다(구매하려는 모델명을 알려주세요)"처럼 답하세요.

너의 출력은 반드시 아래 JSON 형식이어야 한다.

{{
  "answer": string,
  "show_next_recommendation_page": boolean
}}

- **show_next_recommendation_page**: 사용자가 **이미 받은 추천 목록(예: 12개)에서 다음 3개를 보고 싶다**는 의미로 말하면 true.
  - 예: 더 보여줘, 다시 보여줘, 다시 추천해줘, 추천 다시, 다른 걸 추천해줘, 다른 패키지 추천해줘, 다음 추천, 다음 패키지, 다른 조합 보여줘 등 (**새로 AI가 추천을 다시 짜 달라는 뜻이 아니라** 저장된 목록의 **다음 페이지**를 원할 때).
  - 일반 질문(가격, 규격, 배치)이면 false.
- **answer**: show_next_recommendation_page 가 true 이면 짧게 인사 후 **옆(또는 위) 추천 영역에서 다음 옵션을 확인해 달라**는 식으로 1~3문장. false 이면 평소처럼 답변.
- 새 패키지를 지어내지 말 것. JSON 외 텍스트 금지.
"""


def build_rag_prompt_with_package_context(
    user_info: Dict[str, Any],
    user_question: str,
    *,
    package_context: Dict[str, Any],
) -> str:
    """
    RECOMMEND_RAG에서 특정 패키지 상세를 DB로 조회한 결과를 컨텍스트로 포함.
    """
    return f"""
당신은 자취/소형 공간 전문가 AI 어시스턴트입니다.
아래는 지금까지 파악한 사용자 정보와, 사용자가 질문한 '특정 추천 패키지'의 상세 정보입니다.
이 정보를 근거로 질문에 답변하세요. (지어내지 말고, 제공된 데이터 위주로 답하세요)
도면·배치는 이 단계에서 다루지 않습니다.

[사용자 정보]
- 평수: {user_info.get("size")}
- 라이프스타일: {user_info.get("lifestyle")}
- 보유 가전: {user_info.get("owned_appliances")}
- 필요 가전: {user_info.get("needed_appliances")}
- 새 공간 구매 계획(가구/가전): {user_info.get("purchase_plans")}
- 가구/소품 추천 필요: {user_info.get("need_furniture")}
- 가구/소품 요청사항(레거시): {user_info.get("furniture_note")}
- 인테리어 스타일(레거시): {user_info.get("interior_style")}
- 예산 입력: {user_info.get("budget_choice") or user_info.get("budget_manwon")}
- 예산 범위(만원): {user_info.get("budget_range_manwon")}

[패키지 상세(DB 조회 결과)]
{package_context}

[사용자 질문]
{user_question}

추가 규칙(매우 중요):
- 이 단계에서 package_context 안에 'products_details'가 포함되며, 그 안의 각 product의 spec에는 width/height/depth가 들어있을 수 있습니다.
- 사용자가 '크기/치수/가로/세로/높이/깊이/폭/두께'를 물으면 width/height/depth 값을 그대로 인용해 정확한 치수 문장을 만드세요.
- 반드시 제공된 DB 값만 사용하세요. DB 스펙이 비어 있으면 추정하지 말고 "모델 확인 후 정확한 치수를 드릴게요"처럼 답하세요.

너의 출력은 반드시 아래 JSON 형식이어야 한다.
{{
  "answer": string,
  "show_next_recommendation_page": boolean
}}

- **show_next_recommendation_page**: 사용자가 저장된 추천 목록의 **다음 3개(다음 페이지)** 를 보고 싶다는 뜻이면 true (더 보여줘, 다시 추천, 다른 패키지 등 **페이징 의도**). 그 외(이 패키지 질문) false.
- **answer**: true 이면 짧은 확인 멘트 + 옆 추천 영역 안내. false 이면 패키지 상세(DB) 근거로 4~6문장.
- JSON 외 텍스트 금지.
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
- 라이프스타일: {user_info.get("lifestyle")}
- 보유 가전: {user_info.get("owned_appliances")}
- 필요 가전: {user_info.get("needed_appliances")}
- 새 공간 구매 계획(가구/가전): {user_info.get("purchase_plans")}
- 가구/소품 추천 필요: {user_info.get("need_furniture")}
- 가구/소품 요청사항(레거시): {user_info.get("furniture_note")}
- 인테리어 스타일(레거시): {user_info.get("interior_style")}
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


def build_blueprint_rag_prompt(
    user_info: Dict[str, Any],
    *,
    floor_plan: Dict[str, Any],
    selected_package: Dict[str, Any],
    products_details: Optional[Union[Dict[str, Any], List[Any]]] = None,
) -> str:
    """
    BLUEPRINT_RAG 단계: 선택 도면 + 추천 패키지 1개 + 스티커 좌표로 배치 적합성 RAG.
    좌표는 보통 캔버스 대비 정규화(0~1) 또는 픽셀 — 그대로 전달해 모델이 상대 위치를 해석합니다.
    """
    floor_json = json.dumps(floor_plan, ensure_ascii=False)
    pkg_json = json.dumps(selected_package, ensure_ascii=False)
    details_json = json.dumps(products_details or {}, ensure_ascii=False)
    return f"""
당신은 주거 공간의 가전·가구 배치 전문가입니다.
사용자가 **도면 위에 스티커로 배치한 결과**와 **이전에 고른 추천 패키지(제품 목록)** 를 바탕으로,
배수/급수/콘센트/가스 등 **설비 위치**와의 관계가 타당한지, 동선·안전·사용 편의 측면에서 조언을 JSON으로 출력하세요.
지어낸 제품 스펙은 쓰지 말고, 아래 [제품 상세]와 [패키지]에 있는 정보만 인용하세요.

[사용자 프로필(진단)]
- 평수: {user_info.get("size")}
- 라이프스타일: {user_info.get("lifestyle")}
- 예산(참고): {user_info.get("budget_choice") or user_info.get("budget_manwon")} / 범위(만원): {user_info.get("budget_range_manwon")}

[도면·배치 입력]
{floor_json}
- placements: 가전/가구/설비 스티커의 종류와 좌표(정규화 0~1 또는 픽셀 — 프론트가 준 그대로)
- utilities: 콘센트/배수/가스밸브 등 (있으면 반드시 활용)

[선택한 추천 패키지(요약)]
{pkg_json}

[제품 상세(DB, 있으면)]
{details_json}

너의 출력은 반드시 아래 JSON만 포함하세요. 설명 문장은 JSON 밖에 쓰지 마세요.
{{
  "answer": "string, 한국어 4~10문장. 배치 전반 평가와 개선 방향.",
  "placement_warnings": ["string, 문제될 수 있는 배치·동선·설비 거리 등"],
  "utility_notes": ["string, 콘센트/배수/가스 등과의 관계에 대한 코멘트"],
  "suggestions": ["string, 구체적 조정 제안(스티커 이동 방향 등)"]
}}
- placements가 비어 있으면 answer에 먼저 스티커를 배치해 달라고 안내하고 warnings에 간단히 표시하세요.
- 제품명은 과장 없이 필요할 때만 언급합니다.
"""

