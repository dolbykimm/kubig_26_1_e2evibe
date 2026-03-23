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
        return filtered if filtered else ["llama3-8b-8192"]
    except Exception:
        return ["llama3-8b-8192"]  # API 실패 시 폴백


def analyze_personality(학과: str, 학번: str, 이름: str, 자소서: str, model: str) -> str:
    """자소서 텍스트를 받아 외향/내향 성격 요약 문자열을 반환한다."""
    prompt = f"""다음은 지원자의 자기소개서입니다.
학과: {학과} | 학번: {학번} | 이름: {이름}

[자기소개서]
{자소서}

위 내용을 바탕으로 이 지원자의 성격을 **외향형** 또는 **내향형** 중 하나로 판단하고,
그 근거를 2~3문장으로 간결하게 요약해 주세요.
반드시 첫 줄에 "외향형" 또는 "내향형" 한 단어만 적고, 줄바꿈 후 근거를 작성하세요."""

    client = get_groq_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=256,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────
# 성별 추정 (한국 이름 휴리스틱)
# ─────────────────────────────────────────────────────────────
# 이름 마지막 글자 기준 — 겹치는 글자는 별도 처리
_FEMALE = set("지수연은아영희혜선나라미채원빈서하봄별솔유림")
_MALE   = set("준호석훈재기철동혁태강산도일규범찬우")
_AMBIG  = _FEMALE & _MALE  # 겹치는 경우 전체 이름으로 재판단


def estimate_gender(name: str) -> str:
    """한국 이름 마지막 글자·전체 빈도 기반 성별 추정."""
    if not name or len(name) < 2:
        return "미상"
    given = name[1:]          # 성(family name) 제거
    last  = given[-1]

    if last in _FEMALE and last not in _AMBIG:
        return "여"
    if last in _MALE and last not in _AMBIG:
        return "남"

    # 전체 글자 빈도로 재판단
    f = sum(1 for c in given if c in _FEMALE)
    m = sum(1 for c in given if c in _MALE)
    if f > m:
        return "여"
    if m > f:
        return "남"
    return "미상"


# ─────────────────────────────────────────────────────────────
# 자리배치 알고리즘
# ─────────────────────────────────────────────────────────────
def assign_seats(
    df: pd.DataFrame,
    num_people: int,
    personality_mode: str,
    student_id_policy: str,
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
    work["EI"]       = work["성격 판정"].apply(lambda x: "E" if "외향" in str(x) else "I")
    work["성별"]     = work["이름"].apply(estimate_gender)
    work["학번_연도"] = work["학번"].astype(str).str[:2]

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

    # ── 테이블 할당 (E/I offset round-robin) ──────────────────
    tables: list[list[int]] = [[] for _ in range(num_tables)]
    half = num_tables // 2  # I 그룹 시작 오프셋 — E와 엇갈리게

    for pos, idx in enumerate(e_idxs):
        tables[pos % num_tables].append(idx)
    for pos, idx in enumerate(i_idxs):
        tables[(pos + half) % num_tables].append(idx)

    work["테이블_번호"] = 0
    for t_num, members in enumerate(tables):
        for idx in members:
            work.at[idx, "테이블_번호"] = t_num + 1

    return work.sort_values("테이블_번호").reset_index(drop=True)


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


def dept_abbrev_match(short: str, long: str) -> bool:
    """
    학과 약어/줄임말 매칭.
    short 의 각 글자가 long 안에 순서대로 등장하면 True.
    예) "보정관" → "보건정책관리학부"  (보…정…관 순서 확인)
    """
    s = short.strip().replace(" ", "")
    l = long.strip().replace(" ", "")
    if not s or not l:
        return False
    if s == l or s in l:
        return True
    # 순서 보존 부분 수열(subsequence) 검사
    it = iter(l)
    return all(ch in it for ch in s)


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
    위에서부터 순회하다 '모든 열이 빈칸(NaN 또는 공백)'인 첫 행을 찾아
    그 행부터 아래를 모두 제거한다. (하단 평가기준 표 차단)
    """
    for i, (_, row) in enumerate(df.iterrows()):
        if row.apply(lambda v: pd.isna(v) or str(v).strip() == "").all():
            return df.iloc[:i]
    return df


def extract_all_comments(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    모든 시트를 순회하며 (학번, 이름, 학과, 전화번호, 면접_통합데이터) 테이블을 반환한다.

    처리 순서 (시트별)
    ① 동적 헤더 탐색  : 상위 10행 중 ROLE_KEYWORDS 키워드가 가장 많은 행 → 컬럼명으로 설정
    ② 꼬리 자르기     : 모든 열이 빈칸인 첫 행부터 하단 삭제 (평가기준 표 등 제거)
    ③ 식별자 열 ffill : 이름·학번 열의 빈칸 → 위 행 값으로 채움 (병합 셀 대응)
    ④ 데이터 수집     : 메타 열 제외 나머지 전체(숫자·텍스트·점수 불문) →
                        "[컬럼명] 값" 형식으로 이어붙여 면접_통합데이터 생성
    ⑤ 이름 기준 groupby → 여러 행/시트 데이터를 한 사람으로 합산
    """
    all_rows: list[dict] = []

    for sheet_name, df_raw in sheets.items():
        # ① 동적 헤더 탐색
        header_row = _find_header_row(df_raw)
        df = df_raw.iloc[header_row + 1:].copy()
        df.columns = df_raw.iloc[header_row].astype(str).str.strip().tolist()
        df = df.reset_index(drop=True)

        # ② 꼬리 자르기
        df = _trim_tail(df)
        if df.empty:
            continue

        # 열 매핑
        col_map   = resolve_columns(list(df.columns))
        id_col    = col_map.get("학번")
        name_col  = col_map.get("이름")
        dept_col  = col_map.get("학과")
        phone_col = col_map.get("전화번호")

        # ③ 식별자 열 ffill (병합 셀 대응)
        for key_col in [id_col, name_col]:
            if key_col and key_col in df.columns:
                df[key_col] = df[key_col].replace(
                    r"^\s*$", pd.NA, regex=True
                ).ffill()

        meta_cols = {c for c in [id_col, name_col, dept_col, phone_col] if c}
        data_cols = [c for c in df.columns if c not in meta_cols]

        # ④ 행별 데이터 수집
        for _, row in df.iterrows():
            학번 = str(row[id_col]).strip()    if id_col    else ""
            이름 = str(row[name_col]).strip()  if name_col  else ""
            학과 = str(row[dept_col]).strip()  if dept_col  else ""
            전화 = str(row[phone_col]).strip() if phone_col else ""
            if not 이름 and not 학번:
                continue

            # 숫자·점수·텍스트·비고 전부 수집 — 빈 셀·nan만 제외
            text = "\n".join(
                f"[{str(col).strip()}] {str(row[col]).strip()}"
                for col in data_cols
                if pd.notna(row[col])
                and str(row[col]).strip() not in ("", "nan", "NaN", "None")
            )
            if not text:
                continue

            all_rows.append({
                "학번": 학번, "이름": 이름, "학과": 학과, "전화번호": 전화,
                "면접_통합데이터": text,
            })

    if not all_rows:
        return pd.DataFrame(
            columns=["학번", "이름", "학과", "전화번호", "면접_통합데이터"]
        )

    # ⑤ 이름·학번 기준으로 합산 (여러 행/시트에 걸친 데이터 병합)
    return (
        pd.DataFrame(all_rows)
        .groupby(["학번", "이름"], as_index=False)
        .agg({
            "학과":           "first",
            "전화번호":       "first",
            "면접_통합데이터": lambda x: "\n---\n".join(filter(None, x)),
        })
    )


def smart_merge(
    df_base: pd.DataFrame,
    df_comments: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    다중 조건 자동 병합 + 동명이인 감지.

    매칭 우선순위
    1. 학번 완전 일치 → 확정
    2. 이름 일치 + (학과 약어 매칭 OR 전화번호 일치) → 확정
    3. 이름 일치 + 후보 1명뿐 → 확정
    4. 나머지 → ambiguous 목록에 추가

    Returns
    -------
    df_merged  : df_base + 면접_통합데이터 열 (자동 확정분만 채워짐)
    ambiguous  : 수동 해결이 필요한 케이스 목록
        [{"comment_idx": int, "name": str, "comment_text": str,
          "candidates": [{"base_idx": int, "label": str}, ...]}, ...]
    """
    df_merged = df_base.copy()
    df_merged["면접_통합데이터"] = None
    ambiguous: list[dict] = []

    has_phone_in_base = "전화번호" in df_base.columns

    for c_idx, c_row in df_comments.iterrows():
        c_이름  = str(c_row["이름"]).strip()
        c_학번  = str(c_row.get("학번", "")).strip()
        c_학과  = str(c_row.get("학과", "")).strip()
        c_전화  = re.sub(r"\D", "", str(c_row.get("전화번호", "")))
        comment = str(c_row["면접_통합데이터"])

        # 이름이 일치하는 후보 행들
        name_mask = df_base["이름"].astype(str).str.strip() == c_이름
        candidates = df_base[name_mask]

        if candidates.empty:
            continue

        # ── 우선순위 1: 학번 완전 일치 ──────────────────────
        if c_학번:
            exact = candidates[candidates["학번"].astype(str).str.strip() == c_학번]
            if len(exact) == 1:
                df_merged.at[exact.index[0], "면접_통합데이터"] = comment
                continue

        # ── 우선순위 2: 학과 약어 매칭 ──────────────────────
        if c_학과 and len(candidates) > 1:
            dept_match = candidates[
                candidates["학과"].astype(str).apply(
                    lambda d: dept_abbrev_match(c_학과, d) or dept_abbrev_match(d, c_학과)
                )
            ]
            if len(dept_match) == 1:
                df_merged.at[dept_match.index[0], "면접_통합데이터"] = comment
                continue

        # ── 우선순위 2b: 전화번호 일치 ──────────────────────
        if c_전화 and has_phone_in_base and len(candidates) > 1:
            phone_match = candidates[
                candidates["전화번호"].astype(str).apply(
                    lambda p: re.sub(r"\D", "", p) == c_전화
                )
            ]
            if len(phone_match) == 1:
                df_merged.at[phone_match.index[0], "면접_통합데이터"] = comment
                continue

        # ── 우선순위 3: 후보 1명 ────────────────────────────
        if len(candidates) == 1:
            df_merged.at[candidates.index[0], "면접_통합데이터"] = comment
            continue

        # ── 해결 불가 → ambiguous ───────────────────────────
        ambiguous.append({
            "comment_idx":  c_idx,
            "name":         c_이름,
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
    """자소서 기반 판정 + 면접 통합 데이터를 종합해 최종 한 줄로 요약한다."""
    prompt = (
        f"지원자: {이름}\n\n"
        f"[자소서 기반 성격 판정]\n{성격_판정} — {근거}\n\n"
        f"[면접 원시 데이터]\n{str(면접_통합데이터).strip()}\n\n"
        "위 면접 데이터에는 점수, 코멘트, 평가 항목이 섞여 있습니다. "
        "문맥을 파악해 ① 이 지원자의 면접 성적 수준(상위권/중간권/하위권)과 "
        "② 면접관의 주요 평가를 스스로 추출하세요. "
        "그 후 자소서 판정과 종합하여 최종 성격을 판단하되, "
        "면접 성적은 10% 수준의 참고 정보로만 반영하세요.\n"
        "첫 줄에 '외향형' 또는 '내향형'만 쓰고, 줄바꿈 후 한 문장으로 근거를 작성하세요."
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
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="최종_분석")
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
                "아래 각 항목에서 면접표의 이름이 자소서의 어떤 지원자와 연결되는지 선택하세요."
            )

            for case in ambiguous_cases:
                options = ["— 매칭 안 함 —"] + [c["label"] for c in case["candidates"]]
                st.selectbox(
                    f"면접표 **'{case['name']}'** 의 코멘트를 누구와 연결할까요?",
                    options=options,
                    key=f"manual_{case['comment_idx']}",
                )

            if st.button("수동 매칭 확정", type="primary", key="btn_manual_confirm"):
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
                df_final["최종_근거"]     = df_final["근거 요약"]

                targets  = df_final[df_final["면접_통합데이터"].notna()]
                progress = st.progress(0, text="재분석 중...")

                for i, (idx, row) in enumerate(targets.iterrows()):
                    try:
                        result = reanalyze_final(
                            이름=str(row["이름"]).strip(),
                            성격_판정=str(row["성격 판정"]).strip(),
                            근거=str(row["근거 요약"]).strip(),
                            면접_통합데이터=str(row["면접_통합데이터"]).strip(),
                            model=selected_model,
                        )
                        lines = result.splitlines()
                        df_final.at[idx, "최종_성격_판정"] = lines[0].strip()
                        df_final.at[idx, "최종_근거"] = " ".join(
                            l.strip() for l in lines[1:] if l.strip()
                        )
                    except Exception as e:
                        df_final.at[idx, "최종_근거"] = f"오류: {e}"

                    progress.progress((i + 1) / len(targets), text=f"재분석 중... ({i+1}/{len(targets)})")

                progress.empty()

                # 3단계가 참조하는 df_personality를 최종본으로 덮어쓴다
                df_for_stage3 = df_final.copy()
                df_for_stage3["성격 판정"] = df_for_stage3["최종_성격_판정"]
                df_for_stage3["근거 요약"] = df_for_stage3["최종_근거"]
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
                df_seated = assign_seats(df_src, num_people, personality, student_id_policy)

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
                            ei_tag = "🔴E" if r["EI"] == "E" else "🔵I"
                            st.markdown(
                                f"- {gender_icon} {r['이름']} ({r['학과']} / {r['학번_연도']}학번) {ei_tag}"
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
