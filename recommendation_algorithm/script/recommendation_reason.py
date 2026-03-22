"""
recommendation_reason.py
- OpenAI API로 12개 패키지 추천 이유 일괄 생성
- 패키지 정보를 한 번에 넘겨서 JSON 배열로 12개 이유 반환
"""

import json
import os

from openai import OpenAI

# ── 설정 ────────────────────────────────────────────────────────────
# API 키는 환경 변수 OPENAI_API_KEY 로만 주입 (코드/깃에 넣지 말 것)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"


# ================================================================== #
#  패키지 정보 포맷팅
# ================================================================== #

def _format_appliance_info(p: dict) -> dict:
    return {
        "제품명":       p.get("name", ""),
        "카테고리":     p.get("category", ""),
        "정가":         p.get("original_price", 0),
        "할인가":       p.get("price", 0),
        "할인율":       f"{p.get('raw_discount_rate', 0)}%",
        "구독가능":     "가능" if p.get("is_subscribe") else "불가",
        "AI기능":       "있음" if p.get("has_ai") else "없음",
        "에너지등급":   p.get("energy_grade", "정보없음"),
        "프리미엄라인": p.get("premium_line", "일반"),
        "인기도점수":   round(float(p.get("popularity_score") or 0), 1),
    }


def _format_furniture_info(p: dict) -> dict:
    return {
        "제품명":       p.get("name", ""),
        "카테고리":     p.get("category", ""),
        "정가":         p.get("original_price", 0),
        "할인가":       p.get("price", 0),
        "할인율":       f"{p.get('raw_discount_rate', 0)}%",
        "소재등급":     p.get("material_grade", "정보없음"),
        "디자인스타일": p.get("design_style", "정보없음"),
        "친환경":       "해당" if p.get("is_eco_friendly") else "미해당",
        "설치포함":     "포함" if p.get("is_installation_included") else "별도",
    }


ELECTRONICS_CATEGORIES = {
    "TV", "스탠바이미", "냉장고", "전기레인지", "광파오븐/전자레인지",
    "식기세척기", "정수기", "세탁기", "워시타워", "워시콤보",
    "의류관리기", "의류건조기", "청소기", "에어컨", "공기청정기",
    "제습기", "가습기",
}


def _build_package_context(pkg: dict, idx: int, budget: int) -> dict:
    """패키지 1개를 LLM 입력용 딕셔너리로 변환"""
    appliances = []
    furniture  = []

    for p in pkg["products"]:
        cat = p.get("category", "")
        if cat in ELECTRONICS_CATEGORIES:
            appliances.append(_format_appliance_info(p))
        else:
            furniture.append(_format_furniture_info(p))

    total_price  = pkg["total_price"]
    budget_ratio = round(total_price / budget * 100, 1) if budget > 0 else 0

    return {
        "패키지번호": idx + 1,
        "총금액":     total_price,
        "예산대비":   f"{budget_ratio}%",
        "가전":       appliances,
        "가구":       furniture,
    }


# ================================================================== #
#  프롬프트 생성
# ================================================================== #

def _build_prompt(
    packages_context: list,
    starter: str,
    preferences: list,
    budget: int,
    square_footage: int,
) -> tuple:
    """system prompt, user prompt 반환"""

    system_prompt = """당신은 LG 홈스타일 전문 큐레이터입니다.
사용자의 라이프스타일과 선호도를 반영해서 각 추천 패키지에 대한 추천 이유를 작성해주세요.

작성 조건:
- 각 패키지마다 2~3문장으로 작성
- 사용자의 선호 조건과 연결해서 설명
- 가격, 할인, 구독, AI 기능, 에너지 등급, 소재, 디자인 등 제품 정보를 구체적으로 활용
- 친근하고 자연스러운 한국어 톤
- 각 패키지는 독립적으로 작성 (다른 패키지와 비교하거나 언급하지 말 것)
- "패키지 N은", "패키지 N의" 같은 패키지 번호 표현 사용 금지
- 패키지에 "테마" 필드가 있으면 해당 테마(예: 가성비, 프리미엄, 효율 등)의 관점을 추천 이유에 자연스럽게 반영할 것
- 반드시 아래 JSON 형식으로만 응답 (다른 텍스트 없이)

응답 형식:
{"reasons": ["추천이유1", "추천이유2", ..., "추천이유N"]}"""

    preferences_str = ", ".join(preferences) if preferences else "없음"

    user_prompt = f"""사용자 정보:
- 라이프스타일: {starter}
- 주거 환경: {square_footage}평
- 총 예산: {budget:,}원
- 선호 조건: {preferences_str}

추천 패키지 목록:
{json.dumps(packages_context, ensure_ascii=False, indent=2)}

위 {len(packages_context)}개 패키지 각각에 대해 추천 이유를 작성해주세요."""

    return system_prompt, user_prompt


# ================================================================== #
#  OpenAI API 호출
# ================================================================== #

def _call_openai(client, packages_context, starter, preferences, budget, square_footage):
    """OpenAI API 단일 호출 → 추천 이유 리스트 반환"""
    system_prompt, user_prompt = _build_prompt(
        packages_context, starter, preferences, budget, square_footage
    )
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        result  = json.loads(response.choices[0].message.content)
        reasons = result.get("reasons", [])
        while len(reasons) < len(packages_context):
            reasons.append("고객님의 라이프스타일에 맞게 선별된 패키지입니다.")
        return reasons[:len(packages_context)]
    except Exception as e:
        print(f"[recommendation_reason] OpenAI 오류: {e}")
        import traceback
        traceback.print_exc()
        return ["고객님의 라이프스타일에 맞게 선별된 패키지입니다."] * len(packages_context)


def generate_reasons(
    packages: list,
    starter: str,
    preferences: list,
    budget: int,
    square_footage: int,
    themes: list = None,
) -> list:
    """
    테마별로 API를 분리 호출해서 추천 이유 생성
    테마가 있으면 테마별로 그룹핑 후 각각 호출 (총 테마 수만큼 API 호출)
    반환: ["이유1", "이유2", ...]  (packages 순서와 동일)
    """
    from collections import defaultdict

    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY 환경 변수가 없습니다. .env 또는 실행 환경에 설정하세요."
        )
    client = OpenAI(api_key=OPENAI_API_KEY)

    if not themes:
        # 테마 없으면 전체 한 번에 호출
        packages_context = [
            _build_package_context(pkg, i, budget)
            for i, pkg in enumerate(packages)
        ]
        return _call_openai(client, packages_context, starter, preferences, budget, square_footage)

    # 테마별로 그룹핑 {theme: [(원래 index, pkg), ...]}
    theme_groups = defaultdict(list)
    for i, (pkg, theme) in enumerate(zip(packages, themes)):
        theme_groups[theme].append((i, pkg))

    # 결과 저장 (원래 순서 유지)
    reasons = [""] * len(packages)

    for theme, group in theme_groups.items():
        indices = [idx for idx, _ in group]
        pkgs    = [pkg for _, pkg in group]

        # 패키지 컨텍스트 생성 (테마 포함)
        packages_context = []
        for j, pkg in enumerate(pkgs):
            ctx = _build_package_context(pkg, j, budget)
            ctx["테마"] = theme
            packages_context.append(ctx)

        # 테마별 API 호출 (4개씩)
        group_reasons = _call_openai(
            client, packages_context, starter, preferences, budget, square_footage
        )

        # 원래 인덱스에 맞게 삽입
        for idx, reason in zip(indices, group_reasons):
            reasons[idx] = reason

    return reasons