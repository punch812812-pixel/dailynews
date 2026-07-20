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
import json, os, re, sys, datetime, urllib.request, urllib.parse, urllib.error
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
CATEGORIES = ["안전", "기후·환경", "경제", "법·제도", "인권·권리", "과학·기술", "우주",
              "세계", "건강", "노동", "디지털·미디어", "민주주의·정치", "복지", "문화·역사", "동물·생명"]

# 수집 검색어 — 자유롭게 추가·수정 가능
QUERIES = [
    "어린이 정책", "청소년 보호", "교육부 발표", "기후 환경 정책", "폭염 한파 안전",
    "물가 경제", "과학 연구 성과", "우주 발사", "법원 판결", "국제 분쟁 경제",
    "복지 제도 시행", "인권", "동물 보호", "재난 안전 대책", "문화유산",
]

# ── 0단계: 네이버 뉴스 수집 ───────────────────────────
def load_whitelist():
    """신뢰 매체 도메인 목록. 파일이 없으면 필터 없이 경고만 (금요일 실행이 멈추지 않도록)"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source_whitelist.json")
    try:
        return json.load(open(path)).get("domains", [])
    except FileNotFoundError:
        print("   경고: source_whitelist.json이 없어 출처 필터 없이 진행합니다")
        return []

def domain_ok(link, whitelist):
    if not whitelist:
        return True
    host = urllib.parse.urlparse(link).netloc.lower()
    return any(host == d or host.endswith("." + d) for d in whitelist)

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
    whitelist = load_whitelist()
    seen, out, filtered = set(), [], 0
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
            if not domain_ok(link, whitelist):
                filtered += 1
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
    if filtered:
        print(f"   화이트리스트 밖 매체 {filtered}건 제외")
    return out[:cap]

# ── Gemini 호출 ──────────────────────────────────────
_model_list, _model_idx = [], 0

def pick_model():
    """사용 가능 모델을 순위 목록으로 확보. 번호 버전(안정) 우선, 'latest' 별칭은 예비"""
    global _model_list
    if _model_list:
        return _model_list[_model_idx]
    if GEMINI_MODEL:
        _model_list = [GEMINI_MODEL]
        return GEMINI_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_KEY}&pageSize=200"
    with urllib.request.urlopen(url, timeout=60) as r:
        models = json.loads(r.read()).get("models", [])
    names = []
    for m in models:
        name = m.get("name", "").replace("models/", "")
        if "generateContent" not in m.get("supportedGenerationMethods", []):
            continue
        if "flash" not in name:
            continue
        if any(x in name for x in ["image", "tts", "live", "audio", "8b", "lite", "exp", "preview", "thinking"]):
            continue
        names.append(name)
    if not names:
        raise RuntimeError("사용 가능한 flash 모델을 찾지 못했습니다. GEMINI_MODEL 환경변수로 직접 지정하세요.")
    numbered = sorted([n for n in names if "latest" not in n], reverse=True)  # 3.0 > 2.5
    aliases = [n for n in names if "latest" in n]
    _model_list = numbered + aliases
    print(f"   모델 후보: {', '.join(_model_list)}")
    return _model_list[0]

def switch_model():
    """현재 모델이 장애일 때 다음 후보로 전환. 전환했으면 True"""
    global _model_idx
    if _model_idx < len(_model_list) - 1:
        _model_idx += 1
        print(f"   모델 전환 → {_model_list[_model_idx]}")
        return True
    return False

def call_gemini(system, user, max_tokens=16000):
    model = pick_model()
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={GEMINI_KEY}")
    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"maxOutputTokens": max_tokens,
                             "responseMimeType": "application/json"},
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} {e.reason} (모델: {model})")
    try:
        cand = resp["candidates"][0]
        text = cand["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError(f"Gemini 응답 형식 오류: {json.dumps(resp)[:500]}")
    finish = cand.get("finishReason", "STOP")
    if finish != "STOP":
        raise RuntimeError(f"Gemini 응답이 완결되지 않음 (finishReason={finish})")
    return text

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

def with_retry(fn, name, tries=4, wait=20):
    """일시적 API 오류에 자동 재시도. 429(호출 한도)는 분당 한도 리셋을 기다려 75초 대기"""
    import time
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except (ValueError, RuntimeError, json.JSONDecodeError, urllib.error.URLError) as e:
            msg = str(e)
            if "429" in msg:
                delay, note = 75, " — 호출 한도 초과, 75초 대기 후 재시도"
            elif "404" in msg:
                if switch_model():
                    delay, note = 5, " — 이 모델은 호출 불가, 예비 모델로 전환"
                else:
                    delay, note = 5, " — 예비 모델 소진. 저장소 Variables에 GEMINI_MODEL로 모델명을 직접 지정하세요"
            elif any(c in msg for c in ("500", "502", "503", "504")):
                if switch_model():
                    delay, note = 10, " — 서버 장애, 예비 모델로 전환하여 재시도"
                else:
                    delay, note = 90, " — 구글 서버 일시 장애, 90초 대기 후 재시도"
            else:
                delay, note = wait, ""
            print(f"   {name} {attempt}차 시도 실패: {e}{note}")
            if attempt == tries:
                raise
            time.sleep(delay)

# ── 1단계: 선별 ──────────────────────────────────────
SELECTION_SYSTEM = """너는 초등 5~6학년 뉴스 브리핑의 편집자다. 후보 목록에서 정확히 5건을 고른다.
기준: 5~6학년군 성취기준(사회·과학·도덕·실과·체육)에 정확히 연계되고 생각을 여는 기사만.
국내·해외를 섞고, 어린이 생활 직결 기사(안전·또래·어린이 권리)를 최소 1건 포함.
폭력·피해 묘사가 불가피한 소재, 정치적 진영 대립이 중심인 소재는 제외.
같은 사안의 중복 기사는 1건만.
[최근에 이미 다룬 기사] 목록이 주어지면, 같은 사안은 뚜렷한 새 국면(새 결정·새 수치·
새 사건 전개)이 있을 때만 고르고, 지난 내용의 반복이면 제외한다.
출력은 설명 없이 JSON만: {"ids": [후보 id 5개 — 월요일감(생활·안전 등 쉬운 것)부터
금요일감(세계·경제 등 어려운 것) 순서로]}
"""

def select_candidates(candidates, recent_titles):
    cand_text = "\n".join(f"#{c['id']} [{c['pub']}] {c['title']} — {c['desc'][:80]}" for c in candidates)
    recent = "\n".join(f"- {t}" for t in recent_titles) or "- (없음)"
    user = f"[최근에 이미 다룬 기사]\n{recent}\n\n[후보 목록]\n{cand_text}"
    data = parse_json_block(call_gemini(SELECTION_SYSTEM, user, max_tokens=2000))
    ids = [i for i in data.get("ids", []) if isinstance(i, int)][:5]
    if len(ids) < MIN_ARTICLES:
        raise ValueError(f"선별 결과가 {len(ids)}건뿐입니다")
    return ids

# ── 1.5단계: 선정 기사 원문 수집 ─────────────────────
def fetch_article(link, timeout=25):
    """기사 원문 본문 추출. 실패하면 None (요약 기반으로 폴백)"""
    req = urllib.request.Request(link, headers={
        "User-Agent": "Mozilla/5.0 (compatible; NewsBriefBot/1.0; +education)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except Exception:
        return None
    html = None
    for enc in ("utf-8", "euc-kr", "cp949"):
        try:
            html = raw.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if html is None:
        html = raw.decode("utf-8", "ignore")
    # 스크립트·스타일 제거 → article 태그 우선 → 태그 제거 → 짧은 잡음 줄 제거
    html = re.sub(r"<(script|style|noscript)[^>]*>[\s\S]*?</\1>", " ", html, flags=re.I)
    m = re.search(r"<article[^>]*>([\s\S]*?)</article>", html, flags=re.I)
    if m:
        html = m.group(1)
    text = re.sub(r"<br\s*/?>|</p>|</div>", "\n", html, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    for a, b in [("&nbsp;", " "), ("&quot;", '"'), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&#39;", "'")]:
        text = text.replace(a, b)
    lines = [re.sub(r"[ \t]+", " ", l).strip() for l in text.split("\n")]
    body = "\n".join(l for l in lines if len(l) >= 15)
    if len(body) < 300:          # 본문이라 하기엔 너무 짧으면 실패 처리
        return None
    return body[:3500]

# ── 2단계: 집필 ──────────────────────────────────────
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
- 각 기사 400~600자(공백 포함) — 400자 미만은 불합격이다. 각 문단은 3~5문장, 130~200자.
- 정확히 3문단:
  ① 무슨 일이 있었나(사실) ② 왜 그런 일이 생겼나(배경·원리) ③ 우리와 무슨 상관인가(영향·의미)
- 한 문장 30자 내외, 한 문장에 정보 하나. 배경지식을 전제하지 말고 본문 안에서 풀어 설명한다.
- '원문 발췌'가 제공된 기사는 발췌를 주 재료로 쓴다. 발췌에는 광고·기자 소개·메뉴 등
  잡음이 섞여 있을 수 있으니 기사 본문만 골라 쓴다.
- 발췌든 요약이든 원문 문장을 그대로 옮기지 않는다. 연속 15자 이상 동일한 표현이
  나오지 않게 완전히 새로 쓴다.
- 제공된 재료(발췌·요약)에 없는 구체 수치를 지어내지 않는다. 확실한 정보만 쓴다.
- 제목 + 부제 1줄. 부제에는 핵심 사실이나 뜻밖의 지점을 넣는다.
- 모든 문장은 '~다'로 끝나는 평서형으로 쓴다. '~습니다'체를 쓰지 않는다.
- topic은 2~10자의 짧은 주제 라벨이다 (예: "안전 · 기후", "경제", "세계 · 지리").
  기사 제목이나 문장을 topic에 넣지 않는다.

[부속 요소]
- words: 기사당 2~3개 [낱말, 어린이 눈높이 풀이]
- think: 자기 연결형 발문 1개 ("나라면", "우리 반이라면", "직접 찾아보자")
- standards: 기사당 정확히 2개 [코드, 짧은 라벨]. 주 1 + 보조 1.
  제공된 성취기준 목록에 실제로 존재하는 코드만 쓴다. 국어(국) 코드는 쓰지 않는다.
- tags: 기사당 3~4개. 첫 번째 태그는 반드시 다음 대주제 목록에서 하나를 그대로 고른다:
  안전, 기후·환경, 경제, 법·제도, 인권·권리, 과학·기술, 우주, 세계, 건강, 노동,
  디지털·미디어, 민주주의·정치, 복지, 문화·역사, 동물·생명.
  두 번째부터는 기사 고유의 구체 낱말 태그.
- sources: [["언론사명 또는 매체", "<후보의 link 그대로>"]] — 반드시 선택한 후보의 link를 한 글자도 바꾸지 않고 쓴다.
- 각 기사에 "candidate_id": <선택한 후보의 id 숫자> 필드를 포함한다.

[출력]
설명 없이 JSON 하나만 출력:
{"articles": [{"candidate_id": 3, "day": "월", "topic": "...", "scope": "국내|해외",
  "reportDate": "<후보의 pub 값>. 보도", "title": "...", "deck": "...",
  "paras": ["...", "...", "..."], "words": [["...", "..."]], "think": "...",
  "standards": [["6사02-01", "라벨"]], "tags": ["..."], "sources": [["매체명", "https://..."]]}]}
"""

def generate_issue(standards, picked):
    std56 = {k: v for k, v in standards.items() if k.startswith("[6") and not k.startswith("[6국")}
    std_text = "\n".join(f"{k} {v}" for k, v in sorted(std56.items()))
    blocks = []
    for c in picked:
        b = f"#{c['id']} [{c['pub']}] {c['title']}\n   link: {c['link']}\n   요약: {c['desc']}"
        if c.get("fulltext"):
            b += f"\n   원문 발췌:\n{c['fulltext']}"
        blocks.append(b)
    user = (f"다음 {len(picked)}건을 주어진 순서 그대로 월요일부터 기사로 만들어라.\n\n"
            + "\n\n".join(blocks)
            + f"\n\n[사용 가능한 성취기준 목록]\n{std_text}")
    return parse_json_block(call_gemini(GENERATION_SYSTEM, user))

def generate_supplement(standards, candidates, exclude_ids, existing, need):
    """검수 탈락으로 빈 자리를 남은 후보에서 보충 생성"""
    std56 = {k: v for k, v in standards.items() if k.startswith("[6") and not k.startswith("[6국")}
    std_text = "\n".join(f"{k} {v}" for k, v in sorted(std56.items()))
    remain = [c for c in candidates if c["id"] not in exclude_ids]
    cand_text = "\n".join(
        f"#{c['id']} [{c['pub']}] {c['title']}\n   요약: {c['desc']}\n   link: {c['link']}"
        for c in remain)
    covered = "\n".join(f"- {t}" for t in existing)
    user = (f"이미 발행이 확정된 기사 주제:\n{covered}\n\n"
            f"위 주제와 겹치지 않게, 아래 후보에서 정확히 {need}건을 골라 같은 규격으로 만들어라. "
            f"day 값은 임시로 '월'을 쓴다(나중에 재배치됨).\n\n"
            f"[후보 기사 목록]\n{cand_text}\n\n[사용 가능한 성취기준 목록]\n{std_text}")
    return parse_json_block(call_gemini(GENERATION_SYSTEM, user))

def normalize_report_dates(issue, candidates):
    """AI가 쓴 reportDate를 무시하고 후보의 실제 발행일로 덮어쓴다 (형식 오류 원천 차단)"""
    by_id = {c["id"]: c for c in candidates}
    by_link = {c["link"]: c for c in candidates}
    for a in issue.get("articles", []):
        c = by_id.get(a.get("candidate_id"))
        if not c:  # candidate_id가 없으면 출처 링크로 역추적
            c = next((by_link[u] for _n, u in a.get("sources", []) if u in by_link), None)
        if c and c.get("pub"):
            a["reportDate"] = f"{c['pub']} 보도"

REPAIR_SYSTEM = """너는 초등 뉴스 브리핑의 편집자다. 아래 기사들은 본문이 규격(공백 포함 400~600자)보다
짧다. 각 기사의 paras(3문단)에 살을 붙여 전체 450~550자로 보강하라.
규칙: 3문단 구조(사실→배경→상관)와 '~다' 평서형 유지. 함께 제공된 재료(원문 발췌·요약)에
있는 내용만 쓰고 새 수치를 지어내지 않는다. paras 외의 필드는 한 글자도 바꾸지 않는다.
출력은 설명 없이 입력과 같은 형식의 JSON: {"articles": [...]}"""

def repair_length(short_articles, candidates):
    """분량 미달 기사를 재료와 함께 보내 규격 분량으로 확장"""
    by_id = {c["id"]: c for c in candidates}
    by_link = {c["link"]: c for c in candidates}
    blocks = []
    for a in short_articles:
        c = by_id.get(a.get("candidate_id")) or next(
            (by_link[u] for _n, u in a.get("sources", []) if u in by_link), None)
        mat = ""
        if c:
            mat = f"\n[이 기사의 재료]\n요약: {c.get('desc','')}"
            if c.get("fulltext"):
                mat += f"\n원문 발췌:\n{c['fulltext']}"
        blocks.append(json.dumps(a, ensure_ascii=False) + mat)
    user = "\n\n────\n\n".join(blocks)
    return parse_json_block(call_gemini(REPAIR_SYSTEM, user))

# ── 2단계: 기계 검증 ─────────────────────────────────
def validate(issue, standards, candidates):
    cand_links = {c["link"] for c in candidates}
    link_full = {c["link"]: c.get("fulltext") for c in candidates}
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
        if len(a["topic"]) > 14:
            errs.append(f"topic이 너무 김({len(a['topic'])}자): 짧은 주제 라벨이어야 함")
        if len(a["paras"]) != 3:
            errs.append(f"문단 수 {len(a['paras'])} (3이어야 함)")
        body_len = len("".join(a["paras"]).replace("\n", ""))
        if not (BODY_MIN <= body_len <= BODY_MAX):
            errs.append(f"본문 {body_len}자 (허용 {BODY_MIN}~{BODY_MAX})")
        if not (2 <= len(a["words"]) <= 4):
            errs.append(f"낱말 도움 {len(a['words'])}개")
        if len(a["standards"]) != 2:
            errs.append(f"성취기준 {len(a['standards'])}개 (2개여야 함)")
        if not a["tags"] or a["tags"][0] not in CATEGORIES:
            errs.append(f"첫 태그가 대주제 목록에 없음: {a['tags'][:1]}")
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
        # 원문이 있는 기사: 원문 문장을 그대로 옮겼는지 검사 (연속 20자 일치)
        ft = next((link_full.get(u) for _n, u in a["sources"] if link_full.get(u)), None)
        if ft:
            ft_norm = re.sub(r"\s+", "", ft)
            copied = False
            for p in a["paras"]:
                pn = re.sub(r"\s+", "", p)
                for i in range(0, max(1, len(pn) - 20), 10):
                    if pn[i:i + 20] in ft_norm:
                        copied = True
                        break
                if copied:
                    break
            if copied:
                errs.append("원문 문장을 그대로 옮김 (재서술 필요)")
        # 보도일이 발행일(오늘)보다 미래이면 오류 — AI의 날짜 오기 방지
        m = re.search(r"(\d{4})[.\-]\s*(\d{1,2})[.\-]\s*(\d{1,2})", a["reportDate"])
        if not m:
            errs.append(f"보도일 형식 오류: {a['reportDate']}")
        elif datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3))) > datetime.date.today():
            errs.append(f"보도일이 미래 날짜: {a['reportDate']}")
        results.append((a, not errs, errs))
    return results

# ── 3단계: AI 검수 ───────────────────────────────────
REVIEW_SYSTEM = """너는 초등 뉴스 브리핑의 최종 검수자다. 항목은 두 등급으로 나뉜다.

[안전 항목 — 의심스러우면 불합격 (안전 우선)]
A1. 초등 5~6학년이 읽기에 적절한가 (폭력·피해·사망 상세 묘사, 선정성, 공포 조장 없음)
A2. 본문이 특정 정치적 입장을 지지·비난하지 않는가
A3. 함께 제공된 후보 요약에 없는 구체 수치·사실을 본문이 지어내지 않았는가

[규격 항목 — 명백하고 심각한 위반일 때만 불합격, 사소한 어색함은 합격]
B1. 3문단이 대체로 사실→배경→우리와의 상관 흐름인가
B2. 성취기준 태그가 기사 내용과 연결되는가 (전혀 무관할 때만 불합격)
B3. 발문이 어린이가 답할 수 있는 질문인가
B4. 문체가 대체로 '~다' 평서형인가 (전체가 '~습니다'체일 때만 불합격, 한두 문장 혼용은 합격)

기억하라: 규격의 완벽함보다 학생이 매일 읽을거리를 받는 것이 중요하다.
안전 항목 위반이 없다면 웬만하면 합격시켜라.

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
    # 과거 호에서 이미 쓴 기사 URL은 후보에서 제외 (같은 기사 재등장 차단)
    published = {u for iss in issues for a in iss.get("articles", []) for _n, u in a.get("sources", [])}
    before = len(candidates)
    candidates = [c for c in candidates if c["link"] not in published]
    if before - len(candidates):
        print(f"   이미 발행한 기사 {before - len(candidates)}건 제외")
    if len(candidates) < 10:
        sys.exit("결호: 수집된 후보가 너무 적습니다")

    # 최근 4주 발행 제목 — 같은 사안 반복 억제용
    recent_titles = [a["title"] for iss in issues[-4:] for a in iss.get("articles", [])]

    print(f"── 1단계: 제{last_no + 1}호 선별 (Gemini)")
    ids = with_retry(lambda: select_candidates(candidates, recent_titles), "선별")
    by_id = {c["id"]: c for c in candidates}
    picked = [by_id[i] for i in ids if i in by_id]
    print(f"   선정 {len(picked)}건")

    print("── 1.5단계: 선정 기사 원문 수집")
    for c in picked:
        c["fulltext"] = fetch_article(c["link"])
        print(f"   #{c['id']} {'원문 확보' if c['fulltext'] else '실패 → 요약으로 폴백'} — {c['title'][:28]}")

    print(f"── 2단계 전: 집필 (Gemini)")
    issue = with_retry(lambda: generate_issue(standards, picked), "집필")
    normalize_report_dates(issue, candidates)
    print(f"   초안 기사 {len(issue.get('articles', []))}건")

    print("── 2단계: 기계 검증")
    passed, short_only = [], []
    for a, ok, errs in validate(issue, standards, candidates):
        length_only = bool(errs) and all(("본문" in e and "허용" in e) for e in errs)
        mark = "통과" if ok else ("분량 미달 → 보정 대상" if length_only else "탈락")
        print(f"   [{a.get('day', '?')}] {a.get('title', '(제목 없음)')[:30]} → {mark}")
        for e in errs:
            print(f"        · {e}")
        if ok:
            passed.append(a)
        elif length_only:
            short_only.append(a)

    if short_only:
        print(f"── 분량 보정 라운드: {len(short_only)}건 확장 시도")
        try:
            fixed = with_retry(lambda: repair_length(short_only, candidates), "분량 보정")
            normalize_report_dates(fixed, candidates)
            for a, ok, errs in validate(fixed, standards, candidates):
                body_len = len("".join(a.get("paras", [])))
                print(f"   [{a.get('day','?')}] 보정 후 {body_len}자 → {'통과' if ok else '탈락 ' + '; '.join(errs)}")
                if ok:
                    passed.append(a)
        except Exception as e:
            print(f"   분량 보정 실패(해당 기사 제외하고 진행): {e}")
        # 보정 기사가 뒤에 붙어 순서가 섞였으니 요일 난도 순서로 재정렬
        passed.sort(key=lambda a: DAYS.index(a["day"]) if a.get("day") in DAYS else 9)

    final = passed
    if len(passed) >= MIN_ARTICLES:
        print("── 3단계: AI 검수")
        issue["articles"] = passed
        try:
            review = with_retry(lambda: ai_review(issue, candidates), "검수")
        except Exception as e:
            sys.exit(f"결호: AI 검수를 3회 시도에도 완료하지 못했습니다 ({e}). "
                     f"안전을 위해 이번 주는 발행하지 않습니다.")
        verdicts = {r["day"]: r for r in review.get("results", [])}
        final = []
        for a in passed:
            v = verdicts.get(a["day"], {"pass": False, "reasons": ["검수 결과 누락"]})
            print(f"   [{a['day']}] → {'합격' if v['pass'] else '불합격'} {'; '.join(v.get('reasons', []))}")
            if v["pass"]:
                final.append(a)

    # ── 보충 라운드: 5건 미만이면 남은 후보에서 한 번 더 채운다 ──
    if MIN_ARTICLES <= len(final) < 5:
        need = 5 - len(final)
        print(f"── 보충 라운드: {need}건 보충 시도")
        try:
            used = {a.get("candidate_id") for a in issue["articles"]}
            titles = [a["title"] for a in final] + recent_titles
            supp = with_retry(lambda: generate_supplement(standards, candidates, used, titles, need), "보충 생성")
            normalize_report_dates(supp, candidates)
            supp_passed = []
            for a, ok, errs in validate(supp, standards, candidates):
                print(f"   [보충] {a.get('title','?')[:30]} → {'통과' if ok else '탈락'} {errs or ''}")
                if ok:
                    supp_passed.append(a)
            if supp_passed:
                sup_review = with_retry(lambda: ai_review({"articles": supp_passed}, candidates), "보충 검수")
                sup_verdicts = {r["day"]: r for r in sup_review.get("results", [])}
                # 보충분은 day가 겹칠 수 있어 제목으로도 대조하지 않고 순서대로 판정 적용
                for a, r in zip(supp_passed, sup_review.get("results", [])):
                    mark = "합격" if r.get("pass") else "불합격"
                    print(f"   [보충 검수] {a['title'][:30]} → {mark} {'; '.join(r.get('reasons', []))}")
                    if r.get("pass") and len(final) < 5:
                        final.append(a)
        except Exception as e:
            print(f"   보충 라운드 실패(무시하고 진행): {e}")

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
