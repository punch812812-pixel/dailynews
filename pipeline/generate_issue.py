#!/usr/bin/env python3
"""
어린이 사회 뉴스 브리핑 — 주간 자동 발행 파이프라인
매주 금요일 GitHub Actions에서 실행됩니다.

흐름:
  1단계 수집·생성: Claude API(웹 검색)로 한 주 뉴스 선별 → 호(issue) JSON 초안
  2단계 기계 검증: 형식·분량·성취기준 코드 실존·URL 형식 (결정적 검사)
  3단계 AI 검수:   수위·규격 체크리스트 감사 → 탈락 기사 제거
  4단계 발행 판정: 통과 기사 3건 미만이면 결호(스크립트 실패 → 알림), 아니면 issues.json에 추가
"""
import json, os, re, sys, datetime, urllib.request

# ── 설정 ─────────────────────────────────────────────
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-6"
API_URL = "https://api.anthropic.com/v1/messages"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISSUES_PATH = os.path.join(ROOT, "data", "issues.json")
STANDARDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "standards.json")
MIN_ARTICLES = 3          # 이 밑으로 떨어지면 결호
BODY_MIN, BODY_MAX = 350, 700   # 공백 포함 글자 수 허용 범위
DAYS = ["월", "화", "수", "목", "금"]

# ── 공통: API 호출 ────────────────────────────────────
def call_claude(messages, system=None, tools=None, max_tokens=20000):
    body = {"model": MODEL, "max_tokens": max_tokens, "messages": messages}
    if system:
        body["system"] = system
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode(),
        headers={
            "content-type": "application/json",
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())

def response_text(resp):
    return "\n".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")

def parse_json_block(text):
    """응답에서 JSON 오브젝트를 안전하게 추출"""
    text = re.sub(r"```json|```", "", text)
    start = text.find("{")
    if start == -1:
        raise ValueError("응답에 JSON이 없습니다")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("JSON 괄호가 닫히지 않았습니다")

# ── 1단계: 수집·생성 ──────────────────────────────────
GENERATION_SYSTEM = """너는 초등학교 5~6학년을 위한 주간 뉴스 브리핑의 편집자다.
웹 검색으로 최근 7일의 한국·세계 뉴스를 조사하고, 아래 규격에 따라 호(issue) JSON을 만든다.

[선별 기준]
- 2022 개정 교육과정 5~6학년군 성취기준(주로 사회, 필요시 과학·도덕·실과·체육)에 연계되는 "생각을 여는" 기사만 고른다.
- 성취기준에 정확히 꽂히는 기사를 우선한다. 억지 매칭이 필요하면 버린다.
- 국내·해외를 섞고, 어린이 생활과 직결되는 기사(안전, 또래 이슈, 어린이 권리)를 최소 1건 포함한다.
- 좋은 기사가 5건이 안 되면 억지로 채우지 말고 있는 만큼만 만든다.
- 소재 수위: 전쟁·참사·범죄는 다루되 폭력·피해 묘사 없이 구조·제도 중심으로만 쓴다.
  사망 사례의 상세, 가해 방법, 선정적 내용은 쓰지 않는다. 애매하면 그 기사를 제외한다.
- 정치적으로 쟁점인 사안은 특정 입장을 지지하지 않고 사실과 여러 관점을 균형 있게 쓴다.
- 요일 배치: 월(생활·안전 등 쉬운 것) → 금(세계·경제 등 어려운 것)으로 난도 상승.

[본문 규격]
- 각 기사 400~600자(공백 포함), 정확히 3문단:
  ① 무슨 일이 있었나(사실) ② 왜 그런 일이 생겼나(배경·원리) ③ 우리와 무슨 상관인가(영향·의미)
- 한 문장 30자 내외, 한 문장에 정보 하나. 배경지식을 전제하지 말고 본문 안에서 풀어 설명한다.
- 원문을 재서술한다. 원문 문장을 15단어 이상 그대로 옮기지 않는다.
- 제목 + 부제 1줄. 부제에는 핵심 수치나 뜻밖의 사실을 넣는다.
- sources의 URL은 반드시 웹 검색 결과에서 실제로 확인한 URL만 쓴다. URL을 지어내지 않는다.

[부속 요소]
- words: 기사당 2~3개 [낱말, 어린이 눈높이 풀이]
- think: 자기 연결형 발문 1개 ("나라면", "우리 반이라면", "직접 찾아보자")
- standards: 기사당 정확히 2개 [코드, 짧은 라벨]. 주 1개 + 보조 1개.
  아래 제공된 성취기준 목록에 실제로 존재하는 코드만 쓴다. 국어(국) 코드는 쓰지 않는다.
- tags: 기사당 3~4개

[출력]
설명 없이 아래 형식의 JSON 하나만 출력한다:
{
  "no": <직전 호 번호 + 1>,
  "date": "YYYY-MM-DD (오늘 날짜)",
  "dateLabel": "YYYY. M. D.",
  "articles": [ { "day": "월", "topic": "...", "scope": "국내|해외", "reportDate": "YYYY. M. D.(요일) 보도",
     "title": "...", "deck": "...", "paras": ["...", "...", "..."],
     "words": [["...", "..."]], "think": "...",
     "standards": [["6사02-01", "라벨"]], "tags": ["..."], "sources": [["언론사명", "https://..."]] } ]
}
"""

def generate_issue(standards, last_no, today):
    # 성취기준 목록 중 5~6학년군(6으로 시작)만 프롬프트에 포함해 토큰 절약
    std56 = {k: v for k, v in standards.items() if k.startswith("[6") and not k.startswith("[6국")}
    std_text = "\n".join(f"{k} {v}" for k, v in sorted(std56.items()))
    user = (
        f"오늘은 {today.strftime('%Y-%m-%d')} 금요일이다. 직전 호는 제{last_no}호였다.\n"
        f"최근 7일의 뉴스를 웹 검색으로 조사해 제{last_no + 1}호를 만들어라.\n\n"
        f"[사용 가능한 성취기준 목록]\n{std_text}"
    )
    resp = call_claude(
        [{"role": "user", "content": user}],
        system=GENERATION_SYSTEM,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 15}],
    )
    return parse_json_block(response_text(resp))

# ── 2단계: 기계 검증 ──────────────────────────────────
def validate(issue, standards):
    """결정적 검사. 기사별 (통과 여부, 사유 목록) 반환"""
    results = []
    for a in issue.get("articles", []):
        errs = []
        for key in ["day", "topic", "scope", "reportDate", "title", "deck",
                    "paras", "words", "think", "standards", "tags", "sources"]:
            if key not in a:
                errs.append(f"필드 누락: {key}")
        if errs:
            results.append((a, False, errs)); continue
        if a["day"] not in DAYS:
            errs.append(f"요일 값 오류: {a['day']}")
        if a["scope"] not in ("국내", "해외"):
            errs.append(f"scope 값 오류: {a['scope']}")
        if len(a["paras"]) != 3:
            errs.append(f"문단 수 {len(a['paras'])} (3이어야 함)")
        body_len = len("".join(a["paras"]).replace("\n", ""))
        if not (BODY_MIN <= body_len <= BODY_MAX):
            errs.append(f"본문 {body_len}자 (허용 {BODY_MIN}~{BODY_MAX})")
        if not (2 <= len(a["words"]) <= 4):
            errs.append(f"낱말 도움 {len(a['words'])}개")
        if len(a["standards"]) != 2:
            errs.append(f"성취기준 {len(a['standards'])}개 (2개여야 함)")
        for code, _label in a["standards"]:
            key = f"[{code}]" if not code.startswith("[") else code
            if key not in standards:
                errs.append(f"실존하지 않는 성취기준 코드: {code}")
            if "국" in code:
                errs.append(f"국어 성취기준 사용 금지: {code}")
        if not a["sources"] or not all(u.startswith("http") for _n, u in a["sources"]):
            errs.append("출처 URL 형식 오류 또는 없음")
        results.append((a, not errs, errs))
    return results

# ── 3단계: AI 검수 ────────────────────────────────────
REVIEW_SYSTEM = """너는 초등 뉴스 브리핑의 최종 검수자다. 각 기사를 아래 체크리스트로 감사하고,
하나라도 위반이면 그 기사를 불합격 처리한다. 확신이 없으면 불합격으로 처리한다(안전 우선).

체크리스트:
1. 초등 5~6학년이 읽기에 적절한가 (폭력·피해·사망 상세 묘사, 선정성, 공포 조장 없음)
2. 3문단이 사실→배경→우리와의 상관 구조를 따르는가
3. 본문이 특정 정치적 입장을 지지·비난하지 않는가
4. 성취기준 태그가 기사 내용과 실제로 연결되는가 (억지 매칭이면 불합격)
5. 내용에 명백한 사실 왜곡·과장이 없는가 (필요하면 웹 검색으로 핵심 사실 1~2개를 확인)
6. 발문이 자기 연결형이고 어린이가 답할 수 있는 질문인가

설명 없이 JSON만 출력한다:
{"results": [{"day": "월", "pass": true, "reasons": []}, ...]}
"""

def ai_review(issue):
    resp = call_claude(
        [{"role": "user", "content": json.dumps(issue, ensure_ascii=False)}],
        system=REVIEW_SYSTEM,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
    )
    return parse_json_block(response_text(resp))

# ── 4단계: 발행 ───────────────────────────────────────
def main():
    if not API_KEY:
        sys.exit("ANTHROPIC_API_KEY가 설정되지 않았습니다")
    standards = json.load(open(STANDARDS_PATH))
    issues = json.load(open(ISSUES_PATH))
    last_no = max(i["no"] for i in issues) if issues else 0
    today = datetime.date.today()

    print(f"── 1단계: 제{last_no + 1}호 수집·생성")
    issue = generate_issue(standards, last_no, today)
    print(f"   초안 기사 {len(issue.get('articles', []))}건")

    print("── 2단계: 기계 검증")
    passed = []
    for a, ok, errs in validate(issue, standards):
        mark = "통과" if ok else "탈락"
        print(f"   [{a.get('day', '?')}] {a.get('title', '(제목 없음)')[:30]} → {mark}")
        for e in errs:
            print(f"        · {e}")
        if ok:
            passed.append(a)

    if len(passed) >= MIN_ARTICLES:
        print("── 3단계: AI 검수")
        issue["articles"] = passed
        review = ai_review(issue)
        verdicts = {r["day"]: r for r in review.get("results", [])}
        final = []
        for a in passed:
            v = verdicts.get(a["day"], {"pass": False, "reasons": ["검수 결과 누락"]})
            mark = "합격" if v["pass"] else "불합격"
            print(f"   [{a['day']}] → {mark} {'; '.join(v.get('reasons', []))}")
            if v["pass"]:
                final.append(a)
    else:
        final = passed

    print("── 4단계: 발행 판정")
    if len(final) < MIN_ARTICLES:
        sys.exit(f"결호: 통과 기사 {len(final)}건 (< {MIN_ARTICLES}). 이번 주는 발행하지 않습니다.")

    # 요일 재배치 (탈락으로 구멍이 났으면 월부터 다시 채움)
    for i, a in enumerate(final):
        a["day"] = DAYS[i]
    issue["articles"] = final
    issue["no"] = last_no + 1
    issue["date"] = today.strftime("%Y-%m-%d")
    issue["dateLabel"] = f"{today.year}. {today.month}. {today.day}."

    issues.append(issue)
    json.dump(issues, open(ISSUES_PATH, "w"), ensure_ascii=False, indent=2)
    print(f"발행 완료: 제{issue['no']}호 · 기사 {len(final)}건 → data/issues.json")

if __name__ == "__main__":
    main()
