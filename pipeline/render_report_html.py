#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render_report_html.py  (FINAL v6)
- metrics JSON : public/metrics_<YYYYMMDD>_KST.json
- montage PNG  : public/tiles/<YYYYMMDD>/pm25_wind_<YYYYMMDD>_KST_montage.png
- output HTML  : public/reports/report_<YYYYMMDD>_0600.html

업데이트:
  • 몽타주: 테두리 제거, 본문 폭 100% (양쪽 맞춤)
  • 표: 전칸 가운데 정렬, '시간' 열 볼드
  • 불릿(개조체): “국내: … / 동아시아: …” 같은 라벨 금지
      - 첫 줄: 한반도(국내 5도시) 핵심 분석 1줄
      - 둘째 줄: 동아시아 전체(중국+일본 7도시) 핵심 분석 1줄
"""

import os, json
from datetime import datetime, timedelta, timezone
from collections import defaultdict

# .env 로드 (프로젝트 루트의 .env)
try:
    from dotenv import load_dotenv
    import pathlib, os as _os
    root = pathlib.Path(__file__).resolve().parents[1]  # repo 루트(/opt/EnvData_Insight_Studio)
    load_dotenv(root / ".env")
    print(f"[ENV] ENABLE_AI_SECTIONS={_os.getenv('ENABLE_AI_SECTIONS')} "
          f"OPENAI_MODEL={_os.getenv('OPENAI_MODEL')} "
          f"OPENAI_API_KEY={'set' if bool(_os.getenv('OPENAI_API_KEY')) else 'missing'}")
except Exception as _e:
    print(f"[ENV] dotenv load skipped ({_e})")

DOMESTIC = ["서울", "대전", "광주", "부산", "대구"]
CHINA    = ["상하이", "칭다오", "톈진"]
JAPAN    = ["후쿠오카", "오사카", "히로시마"]
GROUPS   = [("국내", DOMESTIC), ("중국", CHINA), ("일본", JAPAN)]

TITLE = "EnvData Insight Studio – Daily Report"
DATA_SOURCE = "ECMWF/CAMS(PM2.5) · NOAA/GFS(풍향, 풍속)"

METRICS_DIR = "public"
TILES_DIR   = "public/tiles"
REPORTS_DIR = "public/reports"

# 경로: FS(실제 파일) / HREF(HTML src) 분리
PUBLIC_DIR = "public"
PUBLIC_TILES_FS = os.path.join(PUBLIC_DIR, "tiles")   # 존재 확인용
TILES_HREF_PREFIX = "../tiles"                        # HTML img src용

def montage_fs_for(ymd_):   # 실제 파일
    return os.path.join(PUBLIC_TILES_FS, ymd_, f"pm25_wind_{ymd_}_KST_montage.png")

def montage_href_for(ymd_): # HTML src
    return f"{TILES_HREF_PREFIX}/{ymd_}/pm25_wind_{ymd_}_KST_montage.png"


# ---------------- util ----------------
# --- AI: sections generation ---
def build_llm_input(date_str, recs):
    """입력 스키마(JSON) 최소 버전 생성 – 시계열만 포함(격자 파생값은 추후 추가 가능)."""
    # 도시 순서/그룹
    cities_order = [c for _, cs in GROUPS for c in cs]
    series = {c: [] for c in cities_order}
    for r in recs:
        if r["city"] in series:
            series[r["city"]].append({"time": r["timestamp_kst"], "pm25": round(float(r["pm25"]), 1)})

    # 피크(국내/동아시아)
    def pick_peak(cities):
        best = (-1, None, None)
        for r in recs:
            if r["city"] in cities:
                v = float(r["pm25"])
                if v > best[0]:
                    best = (v, r["timestamp_kst"], r["city"])
        return {"city": best[2], "time": best[1][-5:], "pm25": int(round(best[0]))}

    payload = {
        "meta": {
            "target_date": date_str,
            "cutoff_hour": 6,
            "timezone": "Asia/Seoul",
            "data_source": "ECMWF/CAMS(PM2.5), NOAA/GFS(10m wind)"
        },
        "cities": {
            "order": cities_order,
            "groups": {"domestic": DOMESTIC, "china": CHINA, "japan": JAPAN}
        },
        "thresholds": {"pm25_standard": 35},
        "timeseries": [{"city": c, "unit": "μg/m³", "values": series[c]} for c in cities_order],
        "derived": {
            "regional_extremes": {
                "domestic_peak":  pick_peak(DOMESTIC),
                "east_asia_peak": pick_peak(CHINA + JAPAN),
            }
        }
    }
    return payload

def call_llm_sections(input_json):
    """
    ENABLE_AI_SECTIONS=true 에서만 호출.
    실패 시 None 반환(렌더러는 기본문 사용).
    """
    import os, json, re
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[LLM] skipped: no OPENAI_API_KEY")
        return None

    # -------- 프롬프트 (부드러운 경어체, 굵게 의무, telegraphic bullets) --------
    system = (
        "You are a Korean environmental forecaster writing a public daily air-quality OUTLOOK (예보). "
        "Write in Korean. Use only <p> and <b> HTML tags. Tone: clear, calm, professional, for general public. "
        "HARD RULES:\n"
        "1) Tense: future only (…되겠습니다/…보이겠습니다/…넘기겠습니다). No past forms, no typos.\n"
        "2) Each of summary and impact is EXACTLY ONE paragraph (<p>…</p>) with 2–3 sentences. "
        "   • summary = daily FLOW (when/where/how strong). Include the 35 μg/m³ standard at least once and put "
        "     ONE key clause in <b>bold</b> (max 1–2 bold tags total).\n"
        "   • impact = CAUSE/MECHANISM (transport, W→E wind component, mixing, topography). Do not repeat sentences/phrases from summary. "
        "     Include ONE concrete anchor (a time, value, or named region) and put ONE key clause in <b>bold</b> (max 1–2 bold tags total).\n"
        "   Numbers per section should be concise (1–2 uses max).\n"
        "3) health = ONE paragraph with 2–4 sentences. Actionable advice (mask for sensitive groups, timing of ventilation/air-purifier, "
        "   check index before activity). Vary sentence endings; do not repeat the same ending twice in a row. Do NOT use awkward future-plan style "
        "   like '…할 예정입니다'. Keep it polite and practical.\n"
        "4) bullets = EXACTLY TWO items in TELEGRAPHIC STYLE (English: telegraphic style; Korean: 개조체). "
        "   That means concise clause-like sentences with a subject and verb, no honorific endings, no advice verbs, no trailing period. "
        "   Examples: '수도권 밤사이 기준 초과 가능성 높음', '산둥 고농도 동진, 서풍 성분 강화로 중부권 단기 영향 예상'. "
        "   Do NOT use labels like '서울 - …', no hyphen lists, no polite endings. "
        "   End each bullet as a telegraphic clause like '… 가능성 높음', '… 영향 예상', '… 일시 상승'. "
        "   No polite endings. No period.\n"

        "5) Use normal spacing; keep a single space after sentence-ending punctuation and after closing tags when followed by text. "
        "   Keep wording natural and not overly stiff.\n"
    )

    user = (
        "Return ONLY the following JSON schema using the input data (city time series + peaks). "
        "SCHEMA:"
        "{"
        "\"meta\":{\"target_date\":string,\"locale\":\"ko-KR\",\"pm25_standard\":number},"
        "\"sections\":{"
        "\"summary\":\"<p>2–3 sentences, daily flow; include 35 μg/m³ once; 1–2 <b>…</b> max</p>\","
        "\"impact\":\"<p>2–3 sentences, cause/mechanism; non-duplicative vs summary; 1–2 <b>…</b> max</p>\","
        "\"health\":{\"html\":\"<p>5–7 sentences; polite future tense; actionable; varied endings; no '…예정입니다'</p>\"},"
        "\"bullets\":[\"telegraphic style #1 (개조체, no honorific)\",\"telegraphic style #2 (개조체, no honorific)\"]"
        "}"
        "}\n"
        "QUALITY GATES:\n"
        "- summary/impact each: exactly one <p>; future tense; include at least one concrete anchor; have 1 bold clause.\n"
        "- bullets: two telegraphic sentences, no polite endings, no periods, no advice verbs.\n"
        "- health: 5–7 sentences, polite future tense, practical, varied endings.\n"
        "INPUT JSON FOLLOWS:\n"
    )

    # -------- SDK 호출 (chat.completions + JSON 모드) --------
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "user", "content": json.dumps(input_json, ensure_ascii=False)}
            ],
        )
        rid = getattr(resp, "id", "n/a")
        print(f"[LLM] ok id={rid}")
        text = resp.choices[0].message.content
        data = json.loads(text)
    except Exception as e:
        print(f"[LLM] error: {repr(e)}")
        return None

    # -------- 후처리 (최소화: bullets만 살짝 정리) --------
    def _fix_bullets(arr):
        """불릿을 '개조체(telegraphic)' 한 줄로 다듬되, 의미 손상 없이 최소 보정."""
        if not isinstance(arr, list):
            return arr
        endings = ("입니다.", "입니다", "합니다.", "합니다", "합니다요", "합니다요.",
                "겠습니다.", "겠습니다", "되겠습니다.", "되겠습니다", "됩니다.", "됩니다",
                "예상됩니다.", "예상됩니다", "예정입니다.", "예정입니다", "요.", "요")
        out = []
        for x in arr[:2]:
            if not isinstance(x, str):
                out.append("")
                continue
            t = x.strip()

            # HTML 태그 제거
            t = re.sub(r"<.*?>", "", t)

            # 라벨/하이픈 패턴만 정리(내용 단어는 건드리지 않음)
            t = t.replace(" - ", " ").replace("—", " ").replace("–", " ").replace(" -", " ").replace("- ", " ")

            # 끝 마침표/느낌표만 제거
            t = t.rstrip(" .!…")

            # 공손어/미래계획체 등 '정확히 일치하는' 끝 어미만 제거(어간은 절대 건드리지 않음)
            for suf in endings:
                if t.endswith(suf):
                    t = t[: -len(suf)].rstrip()
                    break

            out.append(t)
        # 정확히 2개 유지
        return (out + ["", ""])[:2]


    try:
        sec = data.get("sections", {})
        if "bullets" in sec:
            sec["bullets"] = _fix_bullets(sec["bullets"])
    except Exception as e:
        print(f"[LLM] post-fix warn: {repr(e)}")

    return data


    try:
        sec = data.get("sections", {})
        if "summary" in sec: sec["summary"] = _ensure_single_p(sec["summary"])
        if "impact"  in sec: sec["impact"]  = _ensure_single_p(sec["impact"])
        if isinstance(sec.get("health"), dict) and "html" in sec["health"]:
            sec["health"]["html"] = _ensure_single_p(sec["health"]["html"])
        if "bullets" in sec:
            sec["bullets"] = _fix_bullets(sec["bullets"])
    except Exception as e:
        print(f"[LLM] post-fix error: {repr(e)}")

    return data


def now_kst():
    KST = timezone(timedelta(hours=9))
    return datetime.now(KST)

def decide_target_date():
    force = os.getenv("REPORT_FORCE_DATE")
    if force:
        return datetime.strptime(force, "%Y-%m-%d").date()
    t = now_kst()
    return t.date() if t.hour < 6 else (t + timedelta(days=1)).date()

def ymd(d): return d.strftime("%Y%m%d")

def metric_path_for(ymd_):     return os.path.join(METRICS_DIR, f"metrics_{ymd_}_KST.json")
def montage_path_for(ymd_):    return os.path.join(TILES_DIR, ymd_, f"pm25_wind_{ymd_}_KST_montage.png")
def timeseries_path_for(ymd_): return os.path.join(REPORTS_DIR, f"timeseries_{ymd_}_0600.png")


def load_metrics_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_records(arr):
    recs=[]
    for r in arr:
        city=r.get("city"); ts=r.get("timestamp_kst"); v=r.get("pm25")
        if city and ts and v is not None:
            recs.append({"city":city,"timestamp_kst":ts,"pm25":float(v)})
    recs.sort(key=lambda x: datetime.strptime(x["timestamp_kst"],"%Y-%m-%d %H:%M"))
    return recs

def collect_matrix(recs):
    uniq_ts=sorted({r["timestamp_kst"] for r in recs},
                   key=lambda s: datetime.strptime(s,"%Y-%m-%d %H:%M"))
    ordered=[c for _,cs in GROUPS for c in cs]
    by_ts_city=defaultdict(dict)
    for r in recs:
        by_ts_city[r["timestamp_kst"]][r["city"]]=r["pm25"]
    grid=[]; allv=[]
    for ts in uniq_ts:
        row={"time":ts[-5:]}  # HH:MM
        for c in ordered:
            v=by_ts_city.get(ts,{}).get(c)
            if v is None: row[c]=None
            else:
                iv=int(round(v)); row[c]=iv; allv.append(iv)
        grid.append(row)
    return grid, ordered, allv

def top3_threshold(vals):
    if not vals: return None
    s=sorted(vals, reverse=True)
    return s[min(3,len(s))-1]

def html_escape(s):
    return (s.replace("&","&amp;").replace("<","&lt;")
             .replace(">","&gt;").replace('"',"&quot;").replace("'","&#39;"))

def update_manifest(latest_ymd: str, path=os.path.join(REPORTS_DIR, "manifest.json")):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"latest": latest_ymd, "dates": [latest_ymd]}
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                cur = json.load(f)
            dates = list(dict.fromkeys([latest_ymd] + (cur.get("dates") or [])))  # 중복 제거, 최신 맨 앞
            data = {"latest": latest_ymd, "dates": dates}
    except Exception as e:
        print(f"[manifest] read/merge skipped: {e}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] manifest updated -> {path}")

# ---------------- HTML blocks ----------------
def build_table_html(grid, ordered, thr_top):
    head1=['<th class="stub time-col">시간</th>'] + [f"<th>{html_escape(c)}</th>" for c in ordered]
    group_cells=[f'<th class="group" colspan="{len(cs)}">{html_escape(lbl)}</th>' for lbl,cs in GROUPS]

    body=[]
    for r in grid:
        tds=[f'<td class="time-col"><b>{r["time"]}</b></td>']  # 시간열 볼드
        for c in ordered:
            v=r[c]
            if v is None:
                tds.append('<td class="na">—</td>')
                continue
            cls=[]
            if v>=35: cls.append("thresh")                   # 기준 이상 배경
            if thr_top and v>=thr_top: cls.append("top3")    # 상위 3개 진빨강+볼드
            cattr=f' class="{" ".join(cls)}"' if cls else ""
            tds.append(f"<td{cattr}>{v}</td>")
        body.append("<tr>"+"".join(tds)+"</tr>")

    return f"""
<div class="table-wrap">
  <div class="unit-label">(단위: μg/m³)</div>
  <table class="city-table">
    <thead>
      <tr>{"".join(head1)}</tr>
      <tr class="group-row"><th></th>{"".join(group_cells)}</tr>
    </thead>
    <tbody>{''.join(body)}</tbody>
  </table>
</div>
"""

def pick_bullets_domestic_and_ea(grid):
    """개조체 2줄 생성 — 라벨 없이 간결 문장."""
    def hh_label(hhmm): return f"{int(hhmm.split(':')[0])}시"
    # 국내 최대값
    dom_best = None  # (v, time, city)
    for r in grid:
        for c in DOMESTIC:
            v=r.get(c)
            if isinstance(v,int):
                if (dom_best is None) or (v > dom_best[0]):
                    dom_best=(v, r["time"], c)
    # 동아시아(중국+일본) 최대값
    ea_best = None
    for r in grid:
        for c in CHINA+JAPAN:
            v=r.get(c)
            if isinstance(v,int):
                if (ea_best is None) or (v > ea_best[0]):
                    ea_best=(v, r["time"], c)
    # 문장(개조체, 라벨 없음)
    li1 = f"<li>{dom_best[2]} {hh_label(dom_best[1])} 약 <b>{dom_best[0]}</b> — 기준 대비 주의 구간</li>" if dom_best else ""
    li2 = f"<li>{ea_best[2]} {hh_label(ea_best[1])} 약 <b>{ea_best[0]}</b> — 동아시아 최고치</li>" if ea_best else ""
    return f"<ul class='bullets'>{li1}{li2}</ul>"


def build_page_html(date_str, table_html, montage_path=None, bullets_html="", timeseries_img=None,
                    summary_html=None, impact_html=None, health_html=None):
    date_kr=f"{int(date_str[0:4])}년 {int(date_str[5:7])}월 {int(date_str[8:10])}일"
    summary=(f"<p>{date_kr} 낮 동안 국내 대기질은 대체로 낮은 수준을 유지하겠으며, "
              "<b>밤(21시 전후) 수도권 중심으로 기준(35 μg/m³) 이상으로 상승</b>할 가능성이 큽니다. "
              "자정 무렵 정점 형성 후 익일 아침에는 기준 부근으로 내려올 전망입니다. "
              "국외(화북·발해만 일대)의 높은 농도가 밤~새벽에 <b>중부권에 단기 영향을 줄 수 있습니다.</b></p>")
    impact=("<p>낮에는 혼합·확산이 원활해 대체로 낮은 수준이겠습니다. "
            "다만 국외 고농도대가 서해 쪽에 머물 가능성이 있어, 해당 시간대에 수도권·충청에서 기준 이상 구간이 짧게 나타날 수 있습니다. "
            "이 상승은 <b>단기적</b>일 가능성이 높고, 새벽 이후 혼합 회복과 함께 빠르게 완화되겠습니다.</p>")
    health=("<p>오늘 밤부터 내일 아침 사이 수도권·충청은 기준 이상 구간이 일시적으로 나타날 수 있어 "
            "<b>민감군은 KF 마스크</b> 착용을 권합니다. 실내 환기는 짧고 자주가 좋습니다. "
            "영남·호남은 대체로 야외활동에 무리가 없으나, 장시간 외부 활동 전 <b>최신 지수</b> 확인을 권장합니다.</p>")
    if summary_html: summary = summary_html
    if impact_html:  impact  = impact_html
    if health_html:  health  = health_html
    montage = f'<figure class="montage"><img src="{montage_path}" alt="PM2.5 3×3 몽타주"></figure>' if montage_path else ""
    chart_block = (f'<figure class="chart"><img src="{os.path.basename(timeseries_img)}" '
                   f'alt="PM2.5 시계열 그래프"></figure>') if (timeseries_img and os.path.exists(timeseries_img)) else ""
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{TITLE}</title>
<style>
  :root {{
    --bg:#fff; --fg:#111; --muted:#666; --border:#d6dbe6; --grid:#e9edf5;
    --soft:#f5f7fb; --soft2:#f0f3fa; --warn-bg:#ffe9e9; --top3:#c00;
    --viz-width: 860px;
  }}
  body{{ margin:0 auto; max-width:960px; padding:32px 20px;
        font-family:-apple-system,BlinkMacSystemFont,segoe ui,Roboto,Helvetica,Arial,Apple SD Gothic Neo,Malgun Gothic,sans-serif;
        line-height:1.6; color:var(--fg) }}
  header h1{{ margin:0; font-size:22px; font-weight:700 }}
  header .meta{{ margin:4px 0 24px; color:#555; font-size:13px }}
  section h2{{ font-size:18px; margin:24px 0 8px }}
  .bullets{{ margin:8px 0 12px 18px }}
  .bullets li{{ margin:2px 0 }}
  .chart img{{ width:var(--viz-width); max-width:100%; display:block; border:none; margin:0 auto 10px }}
  .montage img{{ width:100%; display:block; border:none; margin:8px 0 12px }}
  .table-wrap{{ position:relative; width:var(--viz-width); max-width:100%; margin:8px auto 18px }}
  .unit-label{{ position:absolute; right:0; top:-20px; font-size:12px; color:#666 }}
  table.city-table{{ width:100%; border-collapse:collapse; table-layout:fixed; text-align:center;
                     font-variant-numeric:tabular-nums; border:1px solid var(--border); background:#fff;
                     border-radius:10px; overflow:hidden }}
  .city-table thead th, .city-table tbody td{{ border-bottom:1px solid var(--grid); border-right:1px solid var(--grid);
                                               padding:6px 8px; font-size:13px; text-align:center }}
  .city-table thead th:last-child, .city-table tbody td:last-child{{ border-right:0 }}
  .city-table thead th{{ background:var(--soft); color:#222; font-weight:600 }}
  .city-table thead .group-row th.group{{ background:var(--soft2); color:#223; text-align:center; font-weight:700 }}
  .city-table td.thresh{{ background:var(--warn-bg) }}
  .city-table td.top3{{ color:var(--top3); font-weight:700 }}
  footer{{ margin-top:28px; color:#666; font-size:12px; text-align:center }}
</style>
</head>
<body>
<header>
  <h1>{TITLE} ({date_kr})</h1>
  <div class="meta">데이터 출처: {DATA_SOURCE}</div>
</header>

<main>
  <section id="summary">
    <h2>1) 종합 요약</h2>
    {summary}
    {montage}
  </section>

  <section id="impact">
    {bullets_html}
    <h2>2) 영향 해석</h2>
    {impact}
  </section>

  <section id="table">
    <h2>3) 도시별 예보</h2>
    {chart_block}
    {table_html}
  </section>

  <section id="health">
    <h2>4) 생활·보건 안내</h2>
    {health}
  </section>
</main>

<footer>© {datetime.now().year} EnvData Insight Studio · Source: ECMWF/CAMS, NOAA/GFS</footer>
</body></html>"""

# ---------------- main ----------------
def main():
    target = decide_target_date()

    y = ymd(target)

    metrics_path   = metric_path_for(y)
    montage_fs     = montage_fs_for(y)     # 존재 확인
    montage_src    = montage_href_for(y)   # HTML src
    timeseries_png = timeseries_path_for(y)

    arr = load_metrics_json(metrics_path)
    recs = normalize_records(arr)
    grid, ordered, allv = collect_matrix(recs)
    thr = top3_threshold(allv)

    # (A) 기본 규칙문 준비
    table_html   = build_table_html(grid, ordered, thr)
    bullets_html = pick_bullets_domestic_and_ea(grid)
    summary_html = None
    impact_html  = None
    health_html  = None

    # (B) AI 사용 토글
    if os.getenv("ENABLE_AI_SECTIONS", "false").lower() in ("1","true","yes"):
        llm_in = build_llm_input(target.strftime("%Y-%m-%d"), recs)
        llm_out = call_llm_sections(llm_in)
        if llm_out and "sections" in llm_out:
            s = llm_out["sections"]
            # 필수 필드만 안전 반영(없으면 기존 규칙문 유지)
            summary_html = s.get("summary")
            impact_html  = s.get("impact")
            health_pack  = s.get("health") or {}
            health_html  = health_pack.get("html")
            # bullets 2개면 교체
            b = s.get("bullets") or []
            if isinstance(b, list) and len(b) == 2:
                bullets_html = "<ul class='bullets'><li>"+b[0]+"</li><li>"+b[1]+"</li></ul>"

    # (C) 최종 HTML 렌더(비어있으면 기본문 사용)
    page_html = build_page_html(
        target.strftime("%Y-%m-%d"),
        table_html,
        montage_path=montage_src if os.path.exists(montage_fs) else None,
        bullets_html=bullets_html,
        timeseries_img=timeseries_png,
        summary_html=summary_html,
        impact_html=impact_html,
        health_html=health_html
    )

    os.makedirs(REPORTS_DIR, exist_ok=True)
    out = os.path.join(REPORTS_DIR, f"report_{y}_0600.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(page_html)
    print(f"[OK] {out}")

    update_manifest(y)

if __name__=="__main__":
    main()
