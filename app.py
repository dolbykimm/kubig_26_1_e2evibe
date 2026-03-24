import io
import math
import os
import re

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from streamlit_gsheets import GSheetsConnection

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Groq client
# ─────────────────────────────────────────────────────────────
_groq_client = None


def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY가 .env 파일에 설정되지 않았습니다.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


@st.cache_data(ttl=300, show_spinner=False)
def fetch_llama_models() -> list[str]:
    """Groq에서 현재 활성 모델 목록을 가져와 'llama' 포함 모델만 반환한다. (5분 캐시)"""
    try:
        client = get_groq_client()
        models = client.models.list().data
        filtered = sorted(
            m.id for m in models if "llama" in m.id.lower()
        )
        return filtered if filtered else ["llama-3.3-70b-versatile"]
    except Exception:
        return ["llama-3.3-70b-versatile"]  # API 실패 시 폴백


def analyze_personality(학과: str, 학번: str, 이름: str, 자소서: str, model: str) -> str:
    """자소서 텍스트를 받아 외향/내향 성격 요약 문자열을 반환한다."""
    prompt = f"""다음은 지원자의 자기소개서입니다.
학과: {학과} | 학번: {학번} | 이름: {이름}

[자기소개서]
{자소서}

이 지원자의 성격을 **외향형** 또는 **내향형** 중 하나로 판단하세요.

[판단 기준 — 반드시 준수]
- 자기소개서는 자기 자신을 좋게 포장하는 글이므로, 스스로 "활발하다", "사교적이다"라고 쓴 표현은 신뢰도를 낮게 보세요.
- **외향형**으로 판단하려면 아래 중 하나 이상이 구체적으로 확인되어야 합니다:
  ① 발표·토론·MC·사회 등 말하는 역할을 자발적으로 맡은 경험
  ② 동아리·학생회·팀 프로젝트 등 대외 활동이 다수이고 주도적 역할 수행
  ③ 낯선 사람과의 적극적 소통·네트워킹 경험이 구체적으로 기술됨
- 위 근거 없이 단순히 긍정적·친화적 서술만 있으면 **내향형**으로 판단하세요.
- 모호하면 내향형으로 판단하세요.

반드시 첫 줄에 "외향형" 또는 "내향형" 한 단어만 적고, 줄바꿈 후 2~3문장으로 근거를 작성하세요."""

    client = get_groq_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=256,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────
# 성별 추정 (LLM 배치 방식)
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def estimate_genders_batch(names_tuple: tuple[str, ...], model: str) -> dict[str, str]:
    """
    이름 목록을 한 번의 LLM 호출로 성별 추정.
    캐시 키로 tuple 사용 (hashable).
    """
    names = list(names_tuple)
    if not names:
        return {}
    names_str = "\n".join(f"- {n}" for n in names)
    prompt = (
        "아래 한국 이름들의 성별을 추정해 반드시 '이름: 남' 또는 '이름: 여' 형식으로만 한 줄씩 답하라.\n"
        f"{names_str}"
    )
    try:
        client = get_groq_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=len(names) * 12 + 30,
        )
        result: dict[str, str] = {}
        for line in resp.choices[0].message.content.strip().split("\n"):
            for name in names:
                if name in line:
                    result[name] = "여" if "여" in line else "남"
                    break
        for name in names:
            result.setdefault(name, "미상")   # 누락된 이름 기본값
        return result
    except Exception:
        return {n: "미상" for n in names}


# ─────────────────────────────────────────────────────────────
# 자리배치 알고리즘
# ─────────────────────────────────────────────────────────────
def assign_seats(
    df: pd.DataFrame,
    num_people: int,
    personality_mode: str,
    student_id_policy: str,
    model: str = "llama-3.3-70b-versatile",
) -> pd.DataFrame:
    """
    규칙
    ① E/I 는 테이블마다 무조건 고루 혼합 (round-robin offset)
    ② 성별도 가능한 한 혼합 (남·여 교차 정렬)
    ③ personality_mode
       - 혼합 배치 : 성별 교차 정렬 후 E/I 분산
       - 유사 배치 : 학과 기준으로 정렬 (같은 학과끼리 가깝게)
       - 무작위    : 무작위 셔플 후 E/I 분산
    ④ student_id_policy
       - 학번 무관          : 학번 무시
       - 동일 학번 분리     : 같은 연도 인원이 다른 테이블로 분산
       - 동일 학번 우선 배치: 같은 연도 인원이 같은 테이블에 집중

    Returns
    -------
    원본 컬럼 + ['성별', 'EI', '학번_연도', '테이블_번호'] 추가된 DataFrame
    (테이블_번호 오름차순 정렬)
    """
    work = df.copy().reset_index(drop=True)
    work["EI"] = work["성격 판정"].apply(lambda x: "E" if "외향" in str(x) else "I")

    # LLM 배치로 성별 추정 (한 번의 API 호출)
    unique_names   = tuple(work["이름"].astype(str).unique())
    gender_map     = estimate_genders_batch(unique_names, model)
    work["성별"]   = work["이름"].astype(str).map(gender_map).fillna("미상")

    work["학번_연도"] = work["학번"].astype(str).str[2:4]

    n          = len(work)
    num_tables = math.ceil(n / num_people)

    # ── 그룹별 정렬 순서 결정 ──────────────────────────────────
    def ordered_indices(mask: pd.Series) -> list[int]:
        sub = work[mask].copy()

        # 1) 학번 정책 선적용
        if student_id_policy == "동일 학번 분리":
            # 연도 오름차순 → round-robin 이 다른 테이블로 퍼뜨림
            sub = sub.sort_values("학번_연도")
        elif student_id_policy == "동일 학번 우선 배치":
            # 연도 내림차순 → 연속 할당으로 같은 테이블에 몰림
            sub = sub.sort_values("학번_연도", ascending=False)

        # 2) 성격 모드 적용
        if personality_mode == "무작위":
            return sub.sample(frac=1).index.tolist()

        if personality_mode == "유사 배치 (비슷한 성격끼리)":
            # 학과 기준 → 같은 학과끼리 인접
            sub = sub.sort_values(["학번_연도", "학과"])
            return sub.index.tolist()

        # 혼합 배치 (default): 남·여 교차 정렬
        males   = sub[sub["성별"] == "남"].index.tolist()
        females = sub[sub["성별"] == "여"].index.tolist()
        others  = sub[sub["성별"] == "미상"].index.tolist()
        merged  = []
        while males or females:
            if males:   merged.append(males.pop(0))
            if females: merged.append(females.pop(0))
        return merged + others

    e_idxs = ordered_indices(work["EI"] == "E")
    i_idxs = ordered_indices(work["EI"] == "I")

    # ── 테이블 할당 (E/I 교대 합산 후 단일 round-robin) ──────
    # E와 I를 교대로 합치면 자연스럽게 각 테이블에 E/I가 섞이고,
    # 단일 round-robin이므로 어떤 테이블도 ceil(n/num_tables) ≤ num_people 를 절대 초과하지 않음
    tables: list[list[int]] = [[] for _ in range(num_tables)]

    merged_idxs: list[int] = []
    for e, i in zip(e_idxs, i_idxs):
        merged_idxs.append(e)
        merged_idxs.append(i)
    merged_idxs.extend(e_idxs[len(i_idxs):])
    merged_idxs.extend(i_idxs[len(e_idxs):])

    for pos, idx in enumerate(merged_idxs):
        tables[pos % num_tables].append(idx)

    work["테이블_번호"] = 0
    for t_num, members in enumerate(tables):
        for idx in members:
            work.at[idx, "테이블_번호"] = t_num + 1

    return work.sort_values("테이블_번호").reset_index(drop=True)


def extract_interview_score(text: str) -> str:
    """면접 통합데이터에서 점수 숫자를 추출해 반환. 못 찾으면 빈 문자열.

    점수는 한 자리 정수(음수 포함)가 대부분이므로
    [총점]/[합계]/[점수] 레이블 뒤에 오는 -?\\d+ 패턴을 우선 찾는다.
    레이블이 없으면 빈 문자열 반환.
    """
    s = str(text)
    m = re.search(
        r"\[(총점|합계|점수|평균|score|total)\]\s*(-?[0-9]+(?:\.[0-9]+)?)",
        s, re.IGNORECASE
    )
    return m.group(2) if m else ""


# ─────────────────────────────────────────────────────────────
# 엑셀 직렬화
# ─────────────────────────────────────────────────────────────
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # 시트1: 테이블별 전체 목록
        df.to_excel(writer, index=False, sheet_name="자리배치_전체")

        # 시트2: 테이블별 요약 (테이블_번호 × 이름·EI·성별)
        summary_rows = []
        for t_num, grp in df.groupby("테이블_번호"):
            members = ", ".join(
                f"{r['이름']}({r['EI']}/{r['성별']})"
                for _, r in grp.iterrows()
            )
            e_cnt = (grp["EI"] == "E").sum()
            i_cnt = (grp["EI"] == "I").sum()
            summary_rows.append({
                "테이블": t_num,
                "인원": len(grp),
                "E수": e_cnt,
                "I수": i_cnt,
                "구성": members,
            })
        pd.DataFrame(summary_rows).to_excel(writer, index=False, sheet_name="테이블_요약")

    return buf.getvalue()


# ─────────────────────────────────────────────────────────────
# 2단계: 면접표 병합 헬퍼
# ─────────────────────────────────────────────────────────────

_SHEET_DETAIL_LIMIT = 10   # 이 개수 이상이면 상세 샘플 생략
_SAMPLE_ROWS        = 3    # 시트당 샘플 행 수
_MAX_COL_WIDTH      = 20   # 셀 값 최대 표시 길이
_MAX_COLS_PER_SHEET = 8    # 시트당 최대 표시 열 수


def build_sheets_summary(sheets: dict[str, pd.DataFrame]) -> str:
    """
    토큰 절약형 시트 구조 요약.
    - 비어 있는 열 제외, 핵심 열만 최대 _MAX_COLS_PER_SHEET 개
    - 셀 값은 _MAX_COL_WIDTH 자로 절단
    - 시트 수 >= _SHEET_DETAIL_LIMIT 이면 시트 이름 목록만 전송
    """
    sheet_names = list(sheets.keys())

    # 시트가 너무 많으면 이름 목록만
    if len(sheets) >= _SHEET_DETAIL_LIMIT:
        return f"시트 목록({len(sheets)}개): {sheet_names}"

    parts = []
    for name, df in sheets.items():
        # 값이 하나라도 있는 열만 추림
        nonempty_cols = [c for c in df.columns if df[c].notna().any()]
        cols = nonempty_cols[:_MAX_COLS_PER_SHEET]

        # 샘플 행: 지정 열만, 셀 값 절단
        sample_df = df[cols].head(_SAMPLE_ROWS).astype(str)
        sample_df = sample_df.applymap(lambda v: v[:_MAX_COL_WIDTH])

        rows_txt = "; ".join(
            " | ".join(f"{c}={sample_df.at[i,c]}" for c in cols)
            for i in sample_df.index
        )
        parts.append(f"[{name}] 열:{cols} / 샘플:{rows_txt}")

    return "\n".join(parts)


def infer_sheet_structure(summary: str, model: str) -> str:
    """Groq LLM에게 시트 분류 기준과 코멘트 추출 방법을 추론하게 한다."""
    prompt = (
        "면접표 엑셀 시트 구조:\n"
        f"{summary}\n\n"
        "질문: 시트 분류 기준(면접관별/날짜별/조별 등)과 코멘트 열 위치를 한국어로 간결히 답하라."
    )

    client = get_groq_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=256,
    )
    return resp.choices[0].message.content.strip()



def _find_header_row(df_raw: pd.DataFrame, max_scan: int = 10) -> int:
    """
    상위 max_scan 행을 순회해 ROLE_KEYWORDS 키워드를 가장 많이 포함한 행 인덱스를 반환한다.
    해당 행이 진짜 컬럼명 행으로 간주된다.
    """
    all_keywords = [kw for kws in ROLE_KEYWORDS.values() for kw in kws]
    best_row, best_count = 0, 0
    for i in range(min(max_scan, len(df_raw))):
        row_vals = df_raw.iloc[i].astype(str).str.lower()
        count = sum(any(kw in cell for kw in all_keywords) for cell in row_vals)
        if count > best_count:
            best_count, best_row = count, i
    return best_row


def _trim_tail(df: pd.DataFrame) -> pd.DataFrame:
    """
    위에서부터 순회해 '모든 열이 빈칸(NaN 또는 공백)'인 첫 행을 찾아
    그 행부터 아래를 모두 제거한다.
    → 데이터 블록 직후의 빈 행이 평가기준 표 등 푸터와의 경계선이 된다.
    """
    for i in range(len(df)):
        if df.iloc[i].apply(lambda v: pd.isna(v) or str(v).strip() == "").all():
            return df.iloc[:i]
    return df


# ─── 지원자 이름 감지 상수 ──────────────────────────────────────
_NAME_PREFIX_RE = re.compile(r"^([가-힣]{2,4})")  # 이름 파싱용 (prefix)
# 한국 성씨 목록 — 이름은 반드시 성씨로 시작한다는 점을 활용해 "활발함" 같은 코멘트 단어를 걸러냄
_KR_SURNAMES    = frozenset(
    "김이박최정강조윤장임한오서신권황안송류전홍고문손양배백허유남심노하곽성차주우구나민"
    "변엄원천방공현함변여추도소석길승라모봉표망제경은편심봉"
)
# 이름 열 헤더 레이블: 이 값만 있는 행은 data_start 후보에서 제외
_HEADER_NAMES   = frozenset({"이름", "성명", "성함", "지원자", "대상자"})


def _is_korean_name(v: str) -> bool:
    """
    셀 값이 한국 이름처럼 보이면 True.
    조건: ① 한글 2~4자로 시작하고 ② 첫 글자가 한국 성씨 목록에 있을 것.
    → "활발함"(활: 非성씨), "논리적"(논: 非성씨) 같은 코멘트 단어를 걸러냄.
    뒤에 "(컴공)", " 20240001" 등이 붙어있어도 OK.
    """
    s = v.strip()
    if s in _HEADER_NAMES or s in ("nan", "NaN", "None", ""):
        return False
    m = _NAME_PREFIX_RE.match(s)
    if not m:
        return False
    name_part = m.group(1)
    return len(name_part) >= 2 and name_part[0] in _KR_SURNAMES


def _find_name_col_idx(df_raw: pd.DataFrame) -> int | None:
    best_col, best_count = None, 0
    for ci in range(len(df_raw.columns)):
        # 이름이 1명만 있는 테스트 시트라도 통과하도록 >= 1로 수정
        cnt = sum(
            1 for v in df_raw.iloc[:, ci].astype(str)
            if _is_korean_name(v) and len(str(v).strip()) < 10
        )
        if cnt > best_count:
            best_count, best_col = cnt, ci
    return best_col if best_count >= 1 else None


def _parse_name_cell(raw: str) -> tuple[str, str, str]:
    """
    이름 셀에서 (순수이름, 학과힌트, 학번힌트)를 분리한다.

    예)  "박지성(컴공)"       → ("박지성", "컴공", "")
         "박지성 20240001"    → ("박지성", "",     "20240001")
         "박지성/통계 2024"   → ("박지성", "통계", "2024")
         "박지성"             → ("박지성", "",     "")
    """
    raw = raw.strip()
    m = _NAME_PREFIX_RE.match(raw)
    if not m:
        return raw, "", ""
    name = m.group(0)
    rest = raw[len(name):].strip().strip("()（）[][]〔〕/|- ")
    # 숫자 연속열(학번)
    id_m     = re.search(r"\d{4,}", rest)
    extra_id = id_m.group(0) if id_m else ""
    # 나머지 한글 부분(학과 힌트)
    extra_dept = re.sub(r"\d+", "", rest).strip("()（）[][]〔〕/|- ").strip()
    return name, extra_dept, extra_id


def extract_all_comments(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    상태 머신 방식으로 면접 통합데이터를 추출한다.

    1. 헤더 동적 탐색: '지원자'/'이름' AND '의견'/'비고' 키워드가 같은 행에 존재하는 행을 헤더로 특정
    2. current_name 상태 유지: 이름 열에 한국어 이름이 나오면 갱신
    3. 의견/비고 열 텍스트를 누적 (빈칸 아닌 모든 셀)
    4. 전체 병합 메모 행 처리: 열 헤더 무관하게 빈칸이 아닌 텍스트 → 현재 추적 지원자에 귀속
    5. '평가기준'/'평가 기준'/'주의사항' 발견 시 즉시 파싱 중단
    """
    _EMPTY = {"", "nan", "NaN", "None", "none"}
    _FOOTER_WORDS = ("평가기준", "평가 기준", "주의사항")
    _NAME_KW  = ("지원자", "이름", "성명", "성함", "대상자")
    _OPINE_KW = ("의견", "비고", "코멘트", "comment", "특이사항", "추가의견")

    def _cell(v) -> str:
        s = str(v).strip()
        return "" if s.lower() in _EMPTY else s

    def _row_has_footer(row_cells: list[str]) -> bool:
        joined = " ".join(row_cells)
        return any(w in joined for w in _FOOTER_WORDS)

    def _find_header_row(df_raw: pd.DataFrame) -> int | None:
        """'이름계열' 키워드 AND '의견계열' 키워드가 동시에 있는 첫 행 인덱스를 반환."""
        for ri in range(min(30, len(df_raw))):
            cells = [_cell(df_raw.iloc[ri, ci]) for ci in range(len(df_raw.columns))]
            cells_lower = [c.lower() for c in cells]
            has_name  = any(any(kw in c for kw in _NAME_KW)  for c in cells_lower)
            has_opine = any(any(kw in c for kw in _OPINE_KW) for c in cells_lower)
            if has_name and has_opine:
                return ri
        return None

    def _find_name_col_from_header(header_cells: list[str]) -> int | None:
        """헤더 행에서 이름 열 인덱스 반환."""
        for ci, c in enumerate(header_cells):
            if any(kw in c.lower() for kw in _NAME_KW):
                return ci
        return None

    def _find_opine_cols(header_cells: list[str]) -> list[int]:
        """헤더 행에서 의견/비고 열 인덱스 목록 반환."""
        return [ci for ci, c in enumerate(header_cells)
                if any(kw in c.lower() for kw in _OPINE_KW)]

    all_rows: list[dict] = []

    for sheet_name, df_raw in sheets.items():
        if df_raw.empty:
            continue

        # ── 1. 헤더 행 탐색 ──────────────────────────────────────────
        header_ri = _find_header_row(df_raw)
        if header_ri is None:
            # 의견 키워드 없어도 이름 열은 있을 수 있으므로 fallback: 이름 열만으로 탐색
            name_ci_fallback = _find_name_col_idx(df_raw)
            if name_ci_fallback is None:
                continue
            # 헤더는 이름이 처음 나오는 행 바로 위
            for ri in range(len(df_raw)):
                if _is_korean_name(_cell(df_raw.iloc[ri, name_ci_fallback])):
                    header_ri = max(0, ri - 1)
                    break
            if header_ri is None:
                continue

        header_cells = [_cell(df_raw.iloc[header_ri, ci]) for ci in range(len(df_raw.columns))]
        name_ci  = _find_name_col_from_header(header_cells)
        opine_cis = _find_opine_cols(header_cells)

        # 이름 열을 헤더에서 못 찾으면 전체 시트에서 재탐색
        if name_ci is None:
            name_ci = _find_name_col_idx(df_raw)
        if name_ci is None:
            continue

        # ── 2. 상태 머신: 헤더 다음 행부터 순회 ─────────────────────
        current_name     = ""
        current_raw_name = ""
        current_id       = ""
        current_dept     = ""
        current_phone    = ""
        current_parts: list[str] = []

        # id/학과/전화 열 인덱스 (헤더에서)
        id_ci_list    = [ci for ci, c in enumerate(header_cells) if "학번" in c or "번호" in c and "전화" not in c]
        dept_ci_list  = [ci for ci, c in enumerate(header_cells) if "학과" in c or "부서" in c or "전공" in c]
        phone_ci_list = [ci for ci, c in enumerate(header_cells) if "전화" in c or "연락" in c]
        id_ci    = id_ci_list[0]    if id_ci_list    else None
        dept_ci  = dept_ci_list[0]  if dept_ci_list  else None
        phone_ci = phone_ci_list[0] if phone_ci_list else None

        def _flush():
            """현재 지원자 데이터를 all_rows 에 저장."""
            if current_name and current_parts:
                all_rows.append({
                    "학번":        current_id,
                    "이름":        current_name,
                    "원본이름":    current_raw_name,
                    "학과":        current_dept,
                    "전화번호":    current_phone,
                    "면접_통합데이터": "\n".join(current_parts),
                })

        stop = False
        for ri in range(header_ri + 1, len(df_raw)):
            row_cells = [_cell(df_raw.iloc[ri, ci]) for ci in range(len(df_raw.columns))]

            # 꼬리 감지 → 즉시 중단
            if _row_has_footer(row_cells):
                stop = True
                break

            # 완전 빈 행 → 건너뜀
            if not any(row_cells):
                continue

            # 이름 열 확인
            name_val = row_cells[name_ci] if name_ci < len(row_cells) else ""
            if _is_korean_name(name_val):
                # 새 지원자 시작 → 이전 지원자 저장
                _flush()
                parsed_name, xdept, xid = _parse_name_cell(name_val)
                current_name     = parsed_name
                current_raw_name = name_val
                current_id       = (row_cells[id_ci] if id_ci is not None and id_ci < len(row_cells) else "") or xid
                current_dept     = (row_cells[dept_ci] if dept_ci is not None and dept_ci < len(row_cells) else "") or xdept
                current_phone    = row_cells[phone_ci] if phone_ci is not None and phone_ci < len(row_cells) else ""
                current_parts    = []

            if not current_name:
                # 아직 지원자 탐색 전 → 건너뜀
                continue

            # ── 의견/비고 열 텍스트 수집 ──────────────────────────────
            if opine_cis:
                for oci in opine_cis:
                    if oci < len(row_cells) and row_cells[oci]:
                        col_label = header_cells[oci] if oci < len(header_cells) else f"col{oci}"
                        current_parts.append(f"[{col_label}] {row_cells[oci]}")
            else:
                # 의견 열이 없으면: 이름/id/학과/전화 외 모든 비어있지 않은 셀 수집
                skip_cis = {c for c in [name_ci, id_ci, dept_ci, phone_ci] if c is not None}
                for ci, val in enumerate(row_cells):
                    if ci not in skip_cis and val:
                        col_label = header_cells[ci] if ci < len(header_cells) else f"col{ci}"
                        current_parts.append(f"[{col_label}] {val}")

            # ── 전체 병합 메모 행 처리 ────────────────────────────────
            # 이름 열이 비어있고 의견 열도 모두 비어있지만 다른 셀에 텍스트 있으면 → 병합 메모
            name_empty = not name_val
            opine_empty = not any(row_cells[oci] for oci in opine_cis if oci < len(row_cells)) if opine_cis else True
            if name_empty and opine_empty:
                skip_cis = {c for c in [name_ci, id_ci, dept_ci, phone_ci] if c is not None}
                memo_vals = [val for ci, val in enumerate(row_cells) if ci not in skip_cis and val]
                if memo_vals:
                    current_parts.append("[메모] " + " / ".join(memo_vals))

        # 마지막 지원자 저장
        if not stop or current_name:
            _flush()

    if not all_rows:
        return pd.DataFrame(columns=["학번", "이름", "원본이름", "학과", "전화번호", "면접_통합데이터"])

    df_res = pd.DataFrame(all_rows)
    df_res["이름"] = df_res["이름"].astype(str).str.strip()
    df_res["학번"] = df_res["학번"].astype(str).str.strip()

    # cross-sheet 동일인 합산: 학번이 있는 레코드만 groupby로 합침.
    # 학번이 없는 레코드는 각각 독립 유지 (동명이인 두 명을 한 명으로 합치는 버그 방지).
    _AGG = {
        "원본이름":        "first",
        "학과":            "first",
        "전화번호":        "first",
        "면접_통합데이터": lambda x: "\n---\n".join(filter(None, x)),
    }
    df_with_id = df_res[df_res["학번"].str.len() > 0]
    df_no_id   = df_res[df_res["학번"].str.len() == 0]

    merged_id = (
        df_with_id.groupby(["학번", "이름"], as_index=False).agg(_AGG)
        if not df_with_id.empty else df_with_id
    )
    return pd.concat([merged_id, df_no_id], ignore_index=True)


def smart_merge(
    df_base: pd.DataFrame,
    df_comments: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    병합 규칙:
    1. 학번 완전 일치 → 자동 확정
    2. 이름 일치 + 후보 1명 → 자동 확정
    3. 동명이인(후보 2명+) → 전원 수동 처리
    """
    df_merged = df_base.copy()
    df_merged["면접_통합데이터"] = None
    ambiguous: list[dict] = []

    for c_idx, c_row in df_comments.iterrows():
        c_이름     = str(c_row["이름"]).strip()
        c_원본이름 = str(c_row.get("원본이름", c_이름)).strip()
        c_학번     = str(c_row.get("학번", "")).strip()
        c_학과     = str(c_row.get("학과", "")).strip()
        comment    = str(c_row["면접_통합데이터"])

        name_mask  = df_base["이름"].astype(str).str.strip() == c_이름
        candidates = df_base[name_mask]

        if candidates.empty:
            continue

        # 학번 완전 일치 → 자동 확정
        if c_학번:
            exact = candidates[candidates["학번"].astype(str).str.strip() == c_학번]
            if len(exact) == 1:
                df_merged.at[exact.index[0], "면접_통합데이터"] = comment
                continue

        # 후보 1명 → 자동 확정
        if len(candidates) == 1:
            df_merged.at[candidates.index[0], "면접_통합데이터"] = comment
            continue

        # 동명이인 → 수동 처리
        ambiguous.append({
            "comment_idx":  c_idx,
            "name":         c_이름,
            "원본이름":     c_원본이름,
            "학과":         c_학과,
            "학번":         c_학번,
            "comment_text": comment,
            "candidates": [
                {
                    "base_idx": idx,
                    "label": (
                        f"{row['이름']} "
                        f"({row.get('학과', '학과?')} / "
                        f"학번: {row.get('학번', '?')})"
                    ),
                }
                for idx, row in candidates.iterrows()
            ],
        })

    return df_merged, ambiguous


def reanalyze_final(
    이름: str,
    성격_판정: str,
    근거: str,
    면접_통합데이터: str,
    model: str,
) -> str:
    """자소서 기반 판정 + 면접 통합 데이터를 종합해 최종 성격을 판단한다."""
    prompt = (
        f"지원자: {이름}\n\n"
        f"[자소서 기반 1차 판정 (참고용)]\n{성격_판정} — {근거}\n\n"
        f"[면접 현장 데이터]\n{str(면접_통합데이터).strip()}\n\n"
        "위 데이터를 바탕으로 다음 기준에 따라 최종 성격을 판단하세요.\n\n"
        "[판단 가중치]\n"
        "• 면접관 코멘트·비고 (가장 중요): '말이 많다', '적극적', '활발', '조용하다', '소극적' 등 직접 관찰 표현을 최우선 근거로 삼으세요.\n"
        "• 면접 점수 (중간 비중): 점수가 높으면 면접 현장에서 자신을 잘 표현했다는 뜻이므로 외향 가능성을 높이는 근거가 됩니다. 단, 점수만으로 외향형을 단정하지는 마세요.\n"
        "• 자소서 1차 판정 (낮은 비중): 면접 데이터와 충돌하면 면접 데이터를 우선하세요.\n\n"
        "[외향형 판정 조건] 아래 중 하나 이상 해당될 때:\n"
        "  ① 면접관이 '활발', '말이 많다', '적극적' 등 긍정적 에너지를 직접 언급\n"
        "  ② 면접 점수가 뚜렷하게 높고(상위권) 코멘트도 긍정적\n"
        "  ③ 두 근거가 모두 약하면 내향형으로 판단하세요.\n\n"
        "다음 형식으로 정확히 3줄만 출력하세요:\n"
        "1줄: '외향형' 또는 '내향형' 한 단어\n"
        "2줄: 이 사람을 대표하는 성격 키워드를 쉼표로 나열 (예: 발표경험 多, 적극적, 리더십) — 20자 이내\n"
        "3줄: 판정 근거 한 문장"
    )

    client = get_groq_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()


def to_final_excel_bytes(df: pd.DataFrame) -> bytes:
    # 원하는 열 순서: 성격판정 → 키워드 → 평균점수 → 근거 → 면접데이터 → 나머지
    preferred = ["최종_성격_판정", "성격_키워드", "면접_평균점수", "최종_근거", "면접_통합데이터"]
    # 실제로 df에 존재하는 열만 preferred 순서로, 없는 열은 끝에 추가
    ordered = [c for c in preferred if c in df.columns]
    rest    = [c for c in df.columns if c not in ordered]
    df_out  = df[rest + ordered]

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False, sheet_name="최종_분석")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────
# Google Sheets 공유
# ─────────────────────────────────────────────────────────────

# session_state 키 → 워크시트 이름 매핑
GS_TARGETS: list[tuple[str, str, str]] = [
    ("자소서 분석 결과", "자소서분석", "df_personality"),
    ("최종 병합 분석",   "최종분석",   "df_merged"),
    ("자리배치 결과",    "자리배치",   "df_seated"),
]


@st.cache_resource(show_spinner=False)
def get_gsheets_conn():
    """GSheetsConnection 캐시 싱글턴. secrets.toml 미설정 시 None 반환."""
    try:
        return st.connection("gsheets", type=GSheetsConnection)
    except Exception:
        return None


def save_to_gsheets(df: pd.DataFrame, worksheet: str) -> None:
    conn = get_gsheets_conn()
    if conn is None:
        raise RuntimeError("Google Sheets 연결 미설정 — .streamlit/secrets.toml을 확인하세요.")
    # 모든 열을 str 변환해 Sheets API 직렬화 오류 방지
    conn.update(worksheet=worksheet, data=df.astype(str))


def load_from_gsheets(worksheet: str) -> pd.DataFrame:
    conn = get_gsheets_conn()
    if conn is None:
        raise RuntimeError("Google Sheets 연결 미설정 — .streamlit/secrets.toml을 확인하세요.")
    return conn.read(worksheet=worksheet, ttl=0)  # ttl=0: 항상 최신 데이터


# ─────────────────────────────────────────────────────────────
# 유연한 열 이름 매핑
# ─────────────────────────────────────────────────────────────
# 각 의미 역할(role)에 대해 엑셀에서 쓰일 법한 동의어를 모두 나열한다.
# 실제 열 이름과 키워드 사이의 '포함 관계'로 점수를 매겨 가장 근접한 열을 선택한다.
ROLE_KEYWORDS: dict[str, list[str]] = {
    "학과":   ["학과", "전공", "계열", "학부", "부서", "소속", "department", "major"],
    "학번":   ["학번", "학생번호", "번호", "student", "id", "학생id", "학생 번호"],
    "이름":   ["이름", "성명", "성함", "name", "지원자", "대상자"],
    "전화번호": ["전화", "연락처", "핸드폰", "휴대폰", "전화번호", "phone", "tel", "mobile"],
}


def _score(col: str, keyword: str) -> int:
    """열 이름과 키워드 사이의 유사도 점수."""
    col_str = str(col) if col is not None else ""
    c = col_str.strip().lower().replace(" ", "")
    k = keyword.strip().lower().replace(" ", "")
    if c == k:
        return 3          # 완전 일치
    if k in c or c in k:
        return 2          # 부분 문자열 포함
    # 글자 단위 겹침 비율 (짧은 쪽 기준)
    common = sum(1 for ch in k if ch in c)
    if len(k) and common / len(k) >= 0.6:
        return 1          # 60 % 이상 글자 겹침
    return 0


def resolve_columns(columns: list[str]) -> dict[str, str | None]:
    """
    엑셀 열 목록 → {role: 실제열이름} 매핑 반환.
    매칭 안 되면 None.
    """
    mapping: dict[str, str | None] = {}
    for role, keywords in ROLE_KEYWORDS.items():
        best_col, best = None, 0
        for col in columns:
            for kw in keywords:
                s = _score(col, kw)
                if s > best:
                    best, best_col = s, col
        mapping[role] = best_col if best > 0 else None
    return mapping


# ─────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="오티 자리배치 앱", layout="wide")
st.title("오티 자리배치 앱")

# ── 사이드바 ─────────────────────────────────────────────────
with st.sidebar:
    # ── 모델 설정 ────────────────────────────────────────────
    st.header("모델 설정")
    with st.spinner("Groq 모델 목록 로딩 중..."):
        llama_models = fetch_llama_models()
    selected_model: str = st.selectbox(
        "사용할 LLM 모델",
        options=llama_models,
        index=0,
        help="Groq에서 현재 활성화된 llama 계열 모델 목록입니다. (5분마다 자동 갱신)",
    )
    st.caption(f"선택된 모델: `{selected_model}`")

    # ── Google Sheets 공유 ───────────────────────────────────
    st.divider()
    st.header("Google Sheets 공유")

    gsheets_ok = get_gsheets_conn() is not None
    if not gsheets_ok:
        st.warning("`secrets.toml` 미설정 — Google Sheets 기능 비활성화")

    for label, ws, ss_key in GS_TARGETS:
        st.markdown(f"**{label}**")
        c_save, c_load = st.columns(2)

        with c_save:
            if st.button(
                "저장",
                key=f"gs_save_{ws}",
                disabled=not (gsheets_ok and ss_key in st.session_state),
                use_container_width=True,
            ):
                try:
                    save_to_gsheets(st.session_state[ss_key], ws)
                    st.success("저장 완료")
                except Exception as e:
                    st.error(str(e))

        with c_load:
            if st.button(
                "불러오기",
                key=f"gs_load_{ws}",
                disabled=not gsheets_ok,
                use_container_width=True,
            ):
                try:
                    df_loaded = load_from_gsheets(ws)
                    st.session_state[ss_key] = df_loaded
                    st.success(f"{len(df_loaded)}행 로드")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

tab1, tab2, tab3 = st.tabs(["1단계: 자소서 분석", "2단계: 면접표 병합", "3단계: 자리배치"])

# ── 1단계: 자소서 분석 ────────────────────────────────────────
with tab1:
    st.header("1단계: 자소서 분석")
    uploaded_resume = st.file_uploader(
        "자소서 엑셀 파일을 업로드하세요",
        type=["xlsx", "xls"],
        key="resume",
    )

    if uploaded_resume is not None:
        df_resume = pd.read_excel(uploaded_resume)
        st.success(f"파일 로드 완료 — {len(df_resume)}행 × {len(df_resume.columns)}열")
        st.dataframe(df_resume, use_container_width=True)

        st.divider()
        st.subheader("열 매핑 확인")
        st.caption("자동으로 감지한 열입니다. 틀린 경우 드롭다운에서 직접 선택하세요.")

        auto_map = resolve_columns(list(df_resume.columns))
        col_options = [None] + list(df_resume.columns)  # None = 미선택

        role_labels = {
            "학과": "학과 / 전공",
            "학번": "학번 / 학생번호",
            "이름": "이름 / 성명",
        }

        # ── 기본 정보 열 3개: selectbox ──────────────────────────
        confirmed_map: dict[str, str | None] = {}
        map_cols = st.columns(3)
        for i, (role, label) in enumerate(role_labels.items()):
            detected = auto_map[role]
            idx = col_options.index(detected) if detected in col_options else 0
            chosen = map_cols[i].selectbox(
                label,
                options=col_options,
                index=idx,
                format_func=lambda x: "— 선택 안 됨 —" if x is None else x,
                key=f"col_map_{role}",
            )
            confirmed_map[role] = chosen

        # ── 자소서 열: 기본 정보 열을 제외한 나머지를 기본 선택 ──
        info_cols = {c for c in confirmed_map.values() if c is not None}
        remaining_cols = [c for c in df_resume.columns if c not in info_cols]

        st.markdown("**자소서 항목 열 선택** (복수 선택 가능 — 선택한 열을 순서대로 이어붙여 LLM에 전달합니다)")
        essay_cols: list[str] = st.multiselect(
            label="자소서 항목 열",
            options=list(df_resume.columns),
            default=remaining_cols,
            key="essay_cols",
            label_visibility="collapsed",
        )

        unresolved = [lbl for role, lbl in role_labels.items() if confirmed_map[role] is None]
        if unresolved:
            st.warning(f"아직 매핑되지 않은 열: **{', '.join(unresolved)}** — 드롭다운에서 선택해 주세요.")
        elif not essay_cols:
            st.warning("자소서 항목 열을 하나 이상 선택해 주세요.")
        else:
            st.divider()
            st.subheader("성격 분석 (Groq · llama3-8b-8192)")

            if st.button("분석 시작", type="primary", key="btn_analyze"):
                results = []
                progress = st.progress(0, text="분석 중...")
                total = len(df_resume)

                for i, row in df_resume.iterrows():
                    # 여러 자소서 열을 "[열이름]\n내용" 형식으로 이어붙임
                    combined_essay = "\n\n".join(
                        f"[{col}]\n{row[col]}"
                        for col in essay_cols
                        if pd.notna(row[col]) and str(row[col]).strip()
                    )
                    try:
                        summary = analyze_personality(
                            학과=str(row[confirmed_map["학과"]]),
                            학번=str(row[confirmed_map["학번"]]),
                            이름=str(row[confirmed_map["이름"]]),
                            자소서=combined_essay,
                            model=selected_model,
                        )
                        lines = summary.splitlines()
                        판정 = lines[0].strip() if lines else "알 수 없음"
                        근거 = " ".join(l.strip() for l in lines[1:] if l.strip())
                    except Exception as e:
                        판정, 근거 = "오류", str(e)

                    results.append({
                        "학과": row[confirmed_map["학과"]],
                        "학번": row[confirmed_map["학번"]],
                        "이름": row[confirmed_map["이름"]],
                        "성격 판정": 판정,
                        "근거 요약": 근거,
                    })
                    progress.progress((i + 1) / total, text=f"분석 중... ({i + 1}/{total})")

                progress.empty()
                df_result = pd.DataFrame(results)
                st.session_state["df_personality"] = df_result
                st.success("분석 완료!")
                st.dataframe(df_result, use_container_width=True)

            elif "df_personality" in st.session_state:
                st.dataframe(st.session_state["df_personality"], use_container_width=True)
    else:
        st.info("엑셀 파일(.xlsx / .xls)을 업로드하면 데이터가 표시됩니다.")

# ── 2단계: 면접표 병합 ────────────────────────────────────────
with tab2:
    st.header("2단계: 면접표 병합")

    uploaded_interview = st.file_uploader(
        "면접표 엑셀 파일을 업로드하세요",
        type=["xlsx", "xls"],
        key="interview",
    )

    if uploaded_interview is not None:
        # ── ① 다중 시트 읽기 ─────────────────────────────────
        sheets: dict[str, pd.DataFrame] = pd.read_excel(
            uploaded_interview, sheet_name=None
        )
        st.success(
            f"파일 로드 완료 — {len(sheets)}개 시트: **{', '.join(sheets.keys())}**"
        )
        for sheet_name, df_s in sheets.items():
            with st.expander(f"시트 미리보기: {sheet_name}  ({len(df_s)}행 × {len(df_s.columns)}열)"):
                st.dataframe(df_s, use_container_width=True)

        st.divider()

        # ── ② LLM 시트 구조 추론 ─────────────────────────────
        st.subheader("시트 구조 자동 추론")
        if st.button("LLM으로 시트 구조 분석", key="btn_infer"):
            with st.spinner("LLM이 시트 구조를 분석 중..."):
                sheet_summary = build_sheets_summary(sheets)
                inference     = infer_sheet_structure(sheet_summary, selected_model)
            st.session_state["sheet_inference"] = inference

        if "sheet_inference" in st.session_state:
            st.info(st.session_state["sheet_inference"])

        st.divider()

        # ── ③ 면접 데이터 추출 + 다중 조건 병합 ──────────────
        st.subheader("면접 데이터 추출 및 1단계 병합")

        has_personality = "df_personality" in st.session_state
        if not has_personality:
            st.warning("1단계 자소서 분석을 먼저 완료해야 병합할 수 있습니다.")

        if st.button("병합 실행", type="primary", disabled=not has_personality, key="btn_merge"):
            with st.spinner("면접 코멘트 추출 중..."):
                df_comments = extract_all_comments(sheets)

            df_base = st.session_state["df_personality"].copy()
            df_merged, ambiguous = smart_merge(df_base, df_comments)

            matched = df_merged["면접_통합데이터"].notna().sum()
            st.session_state["df_merged"]   = df_merged
            st.session_state["ambiguous"]   = ambiguous
            st.session_state["df_comments"] = df_comments

            if ambiguous:
                st.warning(
                    f"자동 매칭 완료 — {matched}/{len(df_merged)}명 | "
                    f"동명이인 **{len(ambiguous)}건** 수동 매칭 필요 ↓"
                )
            else:
                st.success(f"병합 완료 — {matched} / {len(df_merged)}명 면접 코멘트 매칭")

        # ── 동명이인 수동 매칭 구역 ──────────────────────────
        if st.session_state.get("ambiguous"):
            ambiguous_cases = st.session_state["ambiguous"]
            st.divider()
            st.subheader("동명이인 수동 매칭")
            st.caption(
                "같은 이름의 면접 기록이 여러 개입니다. 각 면접 기록을 자소서의 누구와 연결할지 직접 지정하세요."
            )

            # 이름별로 그룹화
            from collections import defaultdict
            groups: dict[str, list[dict]] = defaultdict(list)
            for case in ambiguous_cases:
                groups[case["name"]].append(case)

            for name, cases in groups.items():
                st.markdown(f"### 동명이인: **{name}** ({len(cases)}건)")
                candidates = cases[0]["candidates"]
                cand_options = ["— 매칭 안 함 —"] + [c["label"] for c in candidates]

                for case in cases:
                    # 면접 기록 식별 정보 구성
                    raw  = case.get("원본이름", name)
                    dept = case.get("학과", "")
                    sid  = case.get("학번", "")
                    info_parts = [f"원본: **{raw}**"]
                    if dept: info_parts.append(f"학과: {dept}")
                    if sid:  info_parts.append(f"학번: {sid}")
                    info_str = " / ".join(info_parts)

                    # 코멘트 미리보기 (앞 150자)
                    preview = case["comment_text"][:150].replace("\n", " ")
                    if len(case["comment_text"]) > 150:
                        preview += "…"

                    with st.container(border=True):
                        st.markdown(info_str)
                        if not sid:
                            st.warning("학번 없음 — 원본이름·코멘트 내용으로 직접 구분하세요.")
                        st.caption(preview)
                        st.selectbox(
                            "→ 자소서의 누구와 연결?",
                            options=cand_options,
                            key=f"manual_{case['comment_idx']}",
                        )

                # 같은 이름 그룹 내 중복 배정 경고
                chosen_labels = [
                    st.session_state.get(f"manual_{c['comment_idx']}", "— 매칭 안 함 —")
                    for c in cases
                ]
                non_skip = [l for l in chosen_labels if l != "— 매칭 안 함 —"]
                if len(non_skip) != len(set(non_skip)):
                    st.warning("같은 후보에 두 건 이상 연결되어 있습니다. 확인 후 수정하세요.")

                st.divider()

            if st.button("수동 매칭 확정", type="primary", key="btn_manual_confirm"):
                # 중복 배정 최종 검사
                all_chosen = [
                    st.session_state.get(f"manual_{case['comment_idx']}", "— 매칭 안 함 —")
                    for case in ambiguous_cases
                ]
                non_skip_all = [l for l in all_chosen if l != "— 매칭 안 함 —"]
                if len(non_skip_all) != len(set(non_skip_all)):
                    st.error("동일한 지원자에게 두 개 이상의 면접 기록이 배정되어 있습니다. 수정 후 다시 확정하세요.")
                else:
                    df_m = st.session_state["df_merged"].copy()
                    for case in ambiguous_cases:
                        chosen_label = st.session_state.get(
                            f"manual_{case['comment_idx']}", "— 매칭 안 함 —"
                        )
                        if chosen_label == "— 매칭 안 함 —":
                            continue
                        chosen_idx = next(
                            c["base_idx"]
                            for c in case["candidates"]
                            if c["label"] == chosen_label
                        )
                        df_m.at[chosen_idx, "면접_통합데이터"] = case["comment_text"]

                    st.session_state["df_merged"]  = df_m
                    st.session_state["ambiguous"]  = []
                    matched_final = df_m["면접_통합데이터"].notna().sum()
                    st.success(f"수동 매칭 완료 — 최종 {matched_final}/{len(df_m)}명 매칭")
                    st.rerun()

        if "df_merged" in st.session_state:
            df_merged = st.session_state["df_merged"]
            st.dataframe(df_merged, use_container_width=True)

            st.divider()

            # ── ④ 면접 코멘트 기반 최종 재분석 ─────────────
            st.subheader("면접 코멘트 종합 재분석 (선택)")
            st.caption("자소서 판정 + 면접 코멘트를 함께 LLM에 넘겨 최종 성격을 다시 판정합니다.")

            has_comments = df_merged["면접_통합데이터"].notna().any()
            if not has_comments:
                st.info("매칭된 면접 코멘트가 없어 재분석을 건너뜁니다.")

            if st.button(
                "최종 재분석 실행", type="primary",
                disabled=not has_comments, key="btn_reanalyze"
            ):
                df_final = df_merged.copy()
                df_final["최종_성격_판정"] = df_final["성격 판정"]
                df_final["성격_키워드"]   = ""
                df_final["면접_평균점수"] = ""
                df_final["최종_근거"]     = df_final["근거 요약"]

                targets  = df_final[df_final["면접_통합데이터"].notna()]
                progress = st.progress(0, text="재분석 중...")

                for i, (idx, row) in enumerate(targets.iterrows()):
                    raw_comment = str(row["면접_통합데이터"]).strip()
                    try:
                        result = reanalyze_final(
                            이름=str(row["이름"]).strip(),
                            성격_판정=str(row["성격 판정"]).strip(),
                            근거=str(row["근거 요약"]).strip(),
                            면접_통합데이터=raw_comment,
                            model=selected_model,
                        )
                        lines = [l.strip() for l in result.splitlines() if l.strip()]
                        df_final.at[idx, "최종_성격_판정"] = lines[0] if len(lines) > 0 else ""
                        df_final.at[idx, "성격_키워드"]   = lines[1] if len(lines) > 1 else ""
                        df_final.at[idx, "최종_근거"]     = lines[2] if len(lines) > 2 else (lines[1] if len(lines) > 1 else "")
                    except Exception as e:
                        df_final.at[idx, "최종_근거"] = f"오류: {e}"

                    df_final.at[idx, "면접_평균점수"] = extract_interview_score(raw_comment)
                    progress.progress((i + 1) / len(targets), text=f"재분석 중... ({i+1}/{len(targets)})")

                progress.empty()

                # 3단계가 참조하는 df_personality를 최종본으로 덮어쓴다
                df_for_stage3 = df_final.copy()
                df_for_stage3["성격 판정"] = df_for_stage3["최종_성격_판정"]
                df_for_stage3["근거 요약"] = df_for_stage3["최종_근거"]
                df_for_stage3["성격_키워드"] = df_for_stage3["성격_키워드"]
                st.session_state["df_personality"] = df_for_stage3.drop(
                    columns=["최종_성격_판정", "최종_근거", "면접_통합데이터"], errors="ignore"
                )
                st.session_state["df_merged"] = df_final
                st.success("재분석 완료! 3단계 자리배치에 최종 판정이 자동 반영됩니다.")
                st.dataframe(df_final, use_container_width=True)

            st.divider()

            # ── ⑤ 다운로드 ──────────────────────────────────
            excel_bytes = to_final_excel_bytes(st.session_state["df_merged"])
            st.download_button(
                label="최종 분석 파일 엑셀 다운로드",
                data=excel_bytes,
                file_name="최종_분석_결과.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    else:
        st.info("엑셀 파일(.xlsx / .xls)을 업로드하면 데이터가 표시됩니다.")

# ── 3단계: 자리배치 ───────────────────────────────────────────
with tab3:
    st.header("3단계: 자리배치")

    col_left, col_right = st.columns([1, 2])

    # ── 설정 패널 ────────────────────────────────────────────
    with col_left:
        st.subheader("설정")

        uploaded_txt = st.file_uploader(
            "자리배치 기준 TXT 파일 업로드 (선택)",
            type=["txt"],
            key="seating_txt",
        )
        if uploaded_txt is not None:
            txt_content = uploaded_txt.read().decode("utf-8")
            st.text_area("파일 내용 미리보기", txt_content, height=120)

        st.divider()

        num_people = st.slider("테이블당 인원수", min_value=2, max_value=8, value=4, step=1)

        st.divider()

        personality = st.radio(
            "성격 유형 기준",
            options=["혼합 배치 (외향·내향 섞기)", "유사 배치 (비슷한 성격끼리)", "무작위"],
            index=0,
            help="E/I 혼합은 모든 모드에서 항상 적용됩니다.",
        )

        st.divider()

        student_id_policy = st.radio(
            "학번 기준",
            options=["학번 무관", "동일 학번 분리", "동일 학번 우선 배치"],
            index=0,
        )

    # ── 결과 패널 ────────────────────────────────────────────
    with col_right:
        st.subheader("배치 결과")

        has_data = "df_personality" in st.session_state
        if not has_data:
            st.info("1단계에서 자소서 분석을 먼저 완료해 주세요.")

        if st.button("자리배치 생성", type="primary", disabled=not has_data):
            df_src = st.session_state["df_personality"].copy()

            with st.spinner("배치 중..."):
                df_seated = assign_seats(df_src, num_people, personality, student_id_policy, model=selected_model)

            st.session_state["df_seated"] = df_seated
            st.success(
                f"배치 완료 — 총 {len(df_seated)}명 / "
                f"{df_seated['테이블_번호'].max()}개 테이블"
            )

        if "df_seated" in st.session_state:
            df_seated = st.session_state["df_seated"]
            num_tables = df_seated["테이블_번호"].max()

            # ── 테이블 카드 뷰 ────────────────────────────────
            st.divider()
            st.markdown("#### 테이블별 구성")

            COLS_PER_ROW = 3
            table_nums = sorted(df_seated["테이블_번호"].unique())

            for row_start in range(0, len(table_nums), COLS_PER_ROW):
                card_cols = st.columns(COLS_PER_ROW)
                for col_i, t_num in enumerate(table_nums[row_start: row_start + COLS_PER_ROW]):
                    grp = df_seated[df_seated["테이블_번호"] == t_num]
                    e_cnt = (grp["EI"] == "E").sum()
                    i_cnt = (grp["EI"] == "I").sum()
                    with card_cols[col_i]:
                        st.markdown(f"**테이블 {t_num}** &nbsp; `E:{e_cnt} I:{i_cnt}`")
                        for _, r in grp.iterrows():
                            gender_icon = {"남": "♂", "여": "♀"}.get(r["성별"], "?")
                            ei_color = "#d4a017" if r["EI"] == "E" else "#27ae60"
                            ei_tag = (
                                f'<span style="color:{ei_color};font-weight:bold">'
                                f'{r["EI"]}</span>'
                            )
                            tooltip = str(r.get("성격_키워드", "")).strip()
                            if not tooltip or tooltip in ("nan", "None"):
                                tooltip = ""
                            name_html = (
                                f'<span title="{tooltip}" style="cursor:help;'
                                f'border-bottom:1px dotted #888">{r["이름"]}</span>'
                            )
                            st.markdown(
                                f"- {gender_icon} {name_html} ({r['학과']} / {r['학번_연도']}학번) {ei_tag}",
                                unsafe_allow_html=True,
                            )

            # ── 전체 데이터프레임 ─────────────────────────────
            st.divider()
            st.markdown("#### 전체 목록")
            display_cols = ["테이블_번호", "이름", "학과", "학번", "성별", "EI", "성격 판정"]
            st.dataframe(df_seated[display_cols], use_container_width=True)

            # ── 엑셀 다운로드 ────────────────────────────────
            st.divider()
            excel_bytes = to_excel_bytes(df_seated[display_cols + ["근거 요약"]])
            st.download_button(
                label="엑셀로 다운로드",
                data=excel_bytes,
                file_name="자리배치_결과.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
