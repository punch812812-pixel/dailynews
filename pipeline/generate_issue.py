#!/usr/bin/env python3
"""
어린이 사회 뉴스 브리핑 — 주간 자동 발행 파이프라인 (무료판)
수집: 네이버 뉴스 검색 API (무료) / 생성·검수: Gemini API 무료 등급

흐름:
  0단계 수집:     네이버 뉴스 API로 최근 7일 후보 기사 수집
  1단계 선별·생성: Gemini가 후보 중 5건 선별 → 호(issue) JSON 초안
  2단계 기계 검증: 형식·분량·성취기준 코드 실존·출처 URL이 후보 목록에 실존하는지
  3단계 AI 검수:   수위·규격 체크리스트 감사 → 탈락 기사 제거
  4단계 발행 판정: 통과 3건 미만이면 결호(실패 처리 → 알림), 아니면 issues.json에 추가
"""
import json, os, re, sys, datetime, urllib.request, urllib.parse
from email.utils import parsedate_to_datetime

# ── 설정 ─────────────────────────────────────────────
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
NAVER_ID = os.environ.get("NAVER_CLIENT_ID")
NAVER_SECRET = os.environ.get("NAVER_CLIENT_SECRET")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "")   # 비워 두면 자동 탐지
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISSUES_PATH = os.path.join(ROOT, "data", "issues.json")
STANDARDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "standards.json")
MIN_ARTICLES = 3
BODY_MIN, BODY_MAX = 350, 700
DAYS = ["월", "화", "수", "목", "금"]

# 수집 검색어 — 자유롭게 추가·수정 가능
QUERIES = [
    "어린이 정책", "청소년 보호", "교육부 발표", "기후 환경 정책", "폭염 한파 안전",
    "물가 경제", "과학 연구 성과", "우주 발사", "법원 판결", "국제 분쟁 경제",
    "복지 제도 시행", "인권", "동물 보호", "재난 안전 대책", "문화유산",
]

# ── 0단계: 네이버 뉴스 수집 ───────────────────────────
def strip_tags(s):
    return re.sub(r"<[^>]+>", "", s).replace("&quot;", '"').replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")

def fetch_naver(query, display=20):
    url = "https://openapi.naver.com/v1/search/news.json?" + urllib.parse.urlencode(
        {"query": query, "display": display, "sort": "date"})
    req = urllib.request.Request(url, headers={
        "X-Naver-Client-Id": NAVER_ID, "X-Naver-Client-Secret": NAVER_SECRET})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read()).get("items", [])

def collect_candidates(days=7, cap=60):
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
    seen, out = set(), []
    for q in QUERIES:
        try:
            items = fetch_naver(q)
        except Exception as e:
            print(f"   수집 경고: '{q}' 실패 ({e})")
            continue
        for it in items:
            link = it.get("originallink") or it.get("link", "")
            if not link or link in seen:
                continue
            try:
                pub = parsedate_to_datetime(it["pubDate"])
                if pub < cutoff:
                    continue
                pub_label = pub.strftime("%Y. %-m. %-d.") if os.name != "nt" else pub.strftime("%Y. %m. %d.")
            except Exception:
                pub_label = ""
            seen.add(link)
            out.append({
                "id": len(out) + 1,
                "title": strip_tags(it.get("title", "")),
                "desc": strip_tags(it.get("description", "")),
                "link": link,
                "pub": pub_label,
                "query": q,
            })
    return out[:cap]

# ── Gemini 호출 ──────────────────────────────────────
_model_cache = None

def pick_model():
    """구글 API에서 현재 사용 가능한 모델 목록을 받아 flash 계열 최신 모델을 자동 선택"""
    global _model_cache
    if _model_cache:
        return _model_cache
    if GEMINI_MODEL:                       # 환경변수로 지정했으면 그대로 사용
        _model_cache = GEMINI_MODEL
        return _model_cache
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_KEY}&pageSize=200"
    with urllib.request.urlopen(url, timeout=60) as r:
        models = json.loads(r.read()).get("models", [])
    names = []
    for m in models:
        name = m.get("name", "").replace("models/", "")
        methods = m.get("supportedGenerationMethods", [])
        if "generateContent" not in methods:
            continue
        if "flash" not in name:
            continue
        # 특수 용도(이미지·음성·실시간·경량 8b 등) 제외
        if any(x in name for x in ["image", "tts", "live", "audio", "8b", "lite", "exp", "preview", "thinking"]):
            continue
        names.append(name)
    if not names:
        raise RuntimeError("사용 가능한 flash 모델을 찾지 못했습니다. GEMINI_MODEL 환경변수로 직접 지정하세요.")
    _model_cache = sorted(names)[-1]       # 버전 숫자가 큰 것이 뒤로 정렬됨
    print(f"   사용 모델: {_model_cache}")
    return _model_cache

def call_gemini(system, user, max_tokens=16000):
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{pick_model()}:generateContent?key={GEMINI_KEY}")
    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"maxOutputTokens": max_tokens,
                             "responseMimeType": "application/json"},
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read())
    try:
        return resp["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError(f"Gemini 응답 형식 오류: {json.dumps(resp)[:500]}")

def parse_json_block(text):
    text = re.sub(r"```json|```", "", text)
    start = text.find("{")
    if start == -1:
        raise ValueError("응답에 JSON이 없습니다")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{": depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("JSON 괄호가 닫히지 않았습니다")

# ── 1단계: 선별·생성 ─────────────────────────────────
GENERATION_SYSTEM = """너는 초등학교 5~6학년을 위한 주간 뉴스 브리핑의 편집자다.
제공된 후보 기사 목록에서 5건을 선별하고, 아래 규격에 따라 호(issue) JSON을 만든다.

[선별 기준]
- 2022 개정 교육과정 5~6학년군 성취기준(주로 사회, 필요시 과학·도덕·실과·체육)에 연계되는 "생각을 여는" 기사만 고른다.
- 성취기준에 정확히 꽂히는 기사를 우선한다. 억지 매칭이 필요하면 버린다.
- 국내·해외 소재를 섞고, 어린이 생활과 직결되는 기사(안전, 또래 이슈, 어린이 권리)를 최소 1건 포함한다.
- 적절한 기사가 5건이 안 되면 억지로 채우지 말고 있는 만큼만 만든다.
- 소재 수위: 전쟁·참사·범죄는 다루되 폭력·피해 묘사 없이 구조·제도 중심으로만 쓴다.
  사망 사례의 상세, 가해 방법, 선정적 내용은 쓰지 않는다. 애매하면 그 기사를 제외한다.
- 정치적으로 쟁점인 사안은 특정 입장을 지지하지 않고 사실과 여러 관점을 균형 있게 쓴다.
- 요일 배치: 월(생활·안전 등 쉬운 것) → 금(세계·경제 등 어려운 것)으로 난도 상승.

[본문 규격]
- 각 기사 400~600자(공백 포함), 정확히 3문단:
  ① 무슨 일이 있었나(사실) ② 왜 그런 일이 생겼나(배경·원리) ③ 우리와 무슨 상관인가(영향·의미)
- 한 문장 30자 내외, 한 문장에 정보 하나. 배경지식을 전제하지 말고 본문 안에서 풀어 설명한다.
- 후보의 제목·요약을 재료로 재서술한다. 원문 문장을 그대로 옮기지 않는다.
- 요약에 없는 구체 수치를 지어내지 않는다. 확실한 정보만 쓴다.
- 제목 + 부제 1줄. 부제에는 핵심 사실이나 뜻밖의 지점을 넣는다.

[부속 요소]
- words: 기사당 2~3개 [낱말, 어린이 눈높이 풀이]
- think: 자기 연결형 발문 1개 ("나라면", "우리 반이라면", "직접 찾아보자")
- standards: 기사당 정확히 2개 [코드, 짧은 라벨]. 주 1 + 보조 1.
  제공된 성취기준 목록에 실제로 존재하는 코드만 쓴다. 국어(국) 코드는 쓰지 않는다.
- tags: 기사당 3~4개
- sources: [["언론사명 또는 매체", "<후보의 link 그대로>"]] — 반드시 선택한 후보의 link를 한 글자도 바꾸지 않고 쓴다.
- 각 기사에 "candidate_id": <선택한 후보의 id 숫자> 필드를 포함한다.

[출력]
설명 없이 JSON 하나만 출력:
{"articles": [{"candidate_id": 3, "day": "월", "topic": "...", "scope": "국내|해외",
  "reportDate": "<후보의 pub 값>. 보도", "title": "...", "deck": "...",
  "paras": ["...", "...", "..."], "words": [["...", "..."]], "think": "...",
  "standards": [["6사02-01", "라벨"]], "tags": ["..."], "sources": [["매체명", "https://..."]]}]}
"""

def generate_issue(standards, candidates):
    std56 = {k: v for k, v in standards.items() if k.startswith("[6") and not k.startswith("[6국")}
    std_text = "\n".join(f"{k} {v}" for k, v in sorted(std56.items()))
    cand_text = "\n".join(
        f"#{c['id']} [{c['pub']}] {c['title']}\n   요약: {c['desc']}\n   link: {c['link']}"
        for c in candidates)
    user = (f"[후보 기사 목록]\n{cand_text}\n\n[사용 가능한 성취기준 목록]\n{std_text}")
    return parse_json_block(call_gemini(GENERATION_SYSTEM, user))

# ── 2단계: 기계 검증 ─────────────────────────────────
def validate(issue, standards, candidates):
    cand_links = {c["link"] for c in candidates}
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
        if not a["sources"]:
            errs.append("출처 없음")
        else:
            for _n, u in a["sources"]:
                if u not in cand_links:
                    errs.append("출처 URL이 수집 후보 목록에 없음 (조작·오류 가능)")
        results.append((a, not errs, errs))
    return results

# ── 3단계: AI 검수 ───────────────────────────────────
REVIEW_SYSTEM = """너는 초등 뉴스 브리핑의 최종 검수자다. 각 기사를 아래 체크리스트로 감사하고,
하나라도 위반이면 그 기사를 불합격 처리한다. 확신이 없으면 불합격으로 처리한다(안전 우선).

체크리스트:
1. 초등 5~6학년이 읽기에 적절한가 (폭력·피해·사망 상세 묘사, 선정성, 공포 조장 없음)
2. 3문단이 사실→배경→우리와의 상관 구조를 따르는가
3. 본문이 특정 정치적 입장을 지지·비난하지 않는가
4. 성취기준 태그가 기사 내용과 실제로 연결되는가 (억지 매칭이면 불합격)
5. 함께 제공된 후보 요약에 없는 구체 수치·사실을 본문이 지어내지 않았는가
6. 발문이 자기 연결형이고 어린이가 답할 수 있는 질문인가

설명 없이 JSON만 출력: {"results": [{"day": "월", "pass": true, "reasons": []}]}
"""

def ai_review(issue, candidates):
    by_id = {c["id"]: c for c in candidates}
    ref = "\n".join(
        f"[{a['day']}] 원 후보: {by_id.get(a.get('candidate_id'), {}).get('title', '?')} / "
        f"요약: {by_id.get(a.get('candidate_id'), {}).get('desc', '?')}"
        for a in issue["articles"])
    user = f"[검수 대상]\n{json.dumps(issue, ensure_ascii=False)}\n\n[원 후보 요약 대조용]\n{ref}"
    return parse_json_block(call_gemini(REVIEW_SYSTEM, user))

# ── 4단계: 발행 ──────────────────────────────────────
def main():
    for name, v in [("GEMINI_API_KEY", GEMINI_KEY), ("NAVER_CLIENT_ID", NAVER_ID),
                    ("NAVER_CLIENT_SECRET", NAVER_SECRET)]:
        if not v:
            sys.exit(f"{name}이(가) 설정되지 않았습니다 (저장소 Settings → Secrets)")

    standards = json.load(open(STANDARDS_PATH))
    issues = json.load(open(ISSUES_PATH))
    last_no = max(i["no"] for i in issues) if issues else 0
    today = datetime.date.today()

    print("── 0단계: 네이버 뉴스 수집")
    candidates = collect_candidates()
    print(f"   후보 {len(candidates)}건")
    if len(candidates) < 10:
        sys.exit("결호: 수집된 후보가 너무 적습니다")

    print(f"── 1단계: 제{last_no + 1}호 선별·생성 (Gemini)")
    issue = generate_issue(standards, candidates)
    print(f"   초안 기사 {len(issue.get('articles', []))}건")

    print("── 2단계: 기계 검증")
    passed = []
    for a, ok, errs in validate(issue, standards, candidates):
        print(f"   [{a.get('day', '?')}] {a.get('title', '(제목 없음)')[:30]} → {'통과' if ok else '탈락'}")
        for e in errs:
            print(f"        · {e}")
        if ok:
            passed.append(a)

    final = passed
    if len(passed) >= MIN_ARTICLES:
        print("── 3단계: AI 검수")
        issue["articles"] = passed
        review = ai_review(issue, candidates)
        verdicts = {r["day"]: r for r in review.get("results", [])}
        final = []
        for a in passed:
            v = verdicts.get(a["day"], {"pass": False, "reasons": ["검수 결과 누락"]})
            print(f"   [{a['day']}] → {'합격' if v['pass'] else '불합격'} {'; '.join(v.get('reasons', []))}")
            if v["pass"]:
                final.append(a)

    print("── 4단계: 발행 판정")
    if len(final) < MIN_ARTICLES:
        sys.exit(f"결호: 통과 기사 {len(final)}건 (< {MIN_ARTICLES}). 이번 주는 발행하지 않습니다.")

    for i, a in enumerate(final):
        a["day"] = DAYS[i]
        a.pop("candidate_id", None)
    issues.append({
        "no": last_no + 1,
        "date": today.strftime("%Y-%m-%d"),
        "dateLabel": f"{today.year}. {today.month}. {today.day}.",
        "articles": final,
    })
    json.dump(issues, open(ISSUES_PATH, "w"), ensure_ascii=False, indent=2)
    print(f"발행 완료: 제{last_no + 1}호 · 기사 {len(final)}건 → data/issues.json")

if __name__ == "__main__":
    main()
