# -*- coding: utf-8 -*-
"""설명을 채점한다. 기준은 docs/eval_rubric.md 와 같다.

루브릭이 평가를 둘로 나눈다. VLM 이 맡은 부분은 영상에서 본 사실이 맞는지이고,
규칙이 맡은 부분은 판정에 맞는 표현이 들어갔는지다. 생성 방식 (iii) 은 VLM 출력이
JSON 이고 문장은 규칙이 만들므로 두 부분을 따로 잰다. 같은 자로 재면 과신률이
구조적으로 0 이 되어 (iii) 이 당연히 이긴다.

자동으로 셀 수 있는 것만 센다. 환각의 일부는 사람이 봐야 하므로, 자동 판정과
사람이 볼 항목을 나눠서 내놓는다. 사람이 볼 건수를 줄이는 것이 목적이지
사람을 빼는 것이 아니다.

판정 준수율은 src/vlm/verify.py 의 규칙을 그대로 쓴다. 생성 방식 (ii) 가 쓰는
검사와 같은 코드라야 검사와 채점이 어긋나지 않는다.

    python score_descriptions.py 정답파일.jsonl 출력파일.jsonl [출력파일 ...]

정답 한 줄. land_use 와 verdict 가 채점 기준이 되는 알람이다. 조건 A 와 B 에는
판정이 입력으로 들어가지 않지만, 채점은 같은 이미지에 계산된 판정으로 한다.
    {"image":"a.jpg","class":"흑색연기","flame":false,"area_ratio":0.12,
     "land_use":[{"name":"보전녹지지역","ratio":0.85},
                 {"name":"제1종일반주거지역","ratio":0.15}],
     "verdict":"모호","error_radius_m":55,"risk_targets":["산림","주거지"]}

출력 한 줄. 방식 (i) 은 text 에 원문이, (ii) 와 (iii) 은 description 에 문장이 있다.
    {"image":"a.jpg","model":"qwen","cond":"D","text":"모델이 낸 원문"}
    {"image":"a.jpg","cond":"D","method":"검증재생성","description":"...","attempts":2}
    {"image":"a.jpg","cond":"D","method":"구조화템플릿","description":"...",
     "observed":{"smoke_present":true,...},"schema_ok":true}
"""
import sys, os, json, csv, re
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src", "vlm"))
import verify
from verify import HEDGE          # 유보 표현은 verify.py 한 곳에만 둔다

CLASSES = ["연기", "흑색연기", "백색및회색연기", "백색 및 회색연기", "화염", "불꽃", "불길"]
# 스키마 v0.2 는 영문으로 둔다. 채점은 한국어 기준이므로 바꿔서 쓴다.
CLASS_KO = {"smoke": "연기", "fire": "화염"}
VERDICT_KO = {"confirmed": "확정", "ambiguous": "모호", "undetermined": "미확정"}
SMOKE = ("연기",)
RISK_WORDS = ["산림", "임야", "숲", "주거지", "주택", "아파트", "건물", "도로",
              "농경지", "논", "밭", "시설", "학교", "공장"]

# 구조화 출력(방식 iii)의 사실 항목을 정답과 맞춰 보는 기준이다.
# 라벨 클래스가 색을 하나로 정해 주지 않는 경우가 있어 허용 집합으로 둔다.
COLOR_OK = {"흑색연기": {"검은색"},
            "백색및회색연기": {"회색", "흰색"},
            "백색 및 회색연기": {"회색", "흰색"},
            "연기": {"검은색", "회색", "흰색"}}
# 퍼짐 구간. 라벨 면적 비율로 정한다. 실제 라벨 분포를 보고 다시 맞출 값이다.
SPREAD_BANDS = ((0.05, "좁음"), (0.20, "보통"))
FACT_FIELDS = ["smoke_present", "smoke_color", "spread", "flame_visible"]
FACT_KO = {"smoke_present": "연기유무", "smoke_color": "연기색",
           "spread": "퍼짐", "flame_visible": "화염", "terrain": "지형"}


def spread_of(ratio):
    """면적 비율을 좁음, 보통, 넓음 구간으로 바꾼다."""
    if ratio is None:
        return None
    for limit, name in SPREAD_BANDS:
        if ratio <= limit:
            return name
    return "넓음"


def has_smoke(fact):
    truth = CLASS_KO.get(fact.get("class", ""), fact.get("class", ""))
    return any(k in truth for k in SMOKE)


def alert_of(fact):
    """정답 한 줄을 verify.check 가 받는 알람 모양으로 바꾼다."""
    return {"location_verdict": VERDICT_KO.get(fact.get("verdict"), fact.get("verdict")),
            "position": {"error_radius_m": fact.get("error_radius_m")},
            "land_use": fact.get("land_use") or []}


def parse(text):
    """모델 원문에서 description 을 꺼낸다. 못 꺼내면 형식 미준수다."""
    s = (text or "").strip().strip("`")
    if s.lower().startswith("json"):
        s = s[4:].strip()
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j < 0:
        return None
    try:
        v = json.loads(s[i:j + 1])
    except Exception:
        return None
    if isinstance(v, dict) and isinstance(v.get("description"), str):
        return v["description"].strip()
    return None


def desc_of(rec):
    """출력 한 줄에서 채점할 문장을 꺼낸다.

    방식 (i) 은 모델 원문이 그대로 들어 있어 파싱해야 한다. 방식 (ii) 와 (iii) 은
    실행기가 이미 문장만 뽑아 두었으므로 그대로 쓴다.
    """
    if "description" in rec:
        d = rec.get("description")
        return (d.strip() if isinstance(d, str) and d.strip() else None)
    return parse(rec.get("text", ""))


def score_facts(observed, fact):
    """구조화 출력의 사실 항목이 맞았는지 센다. 정답이 없는 항목은 None 이다."""
    if not isinstance(observed, dict):
        return {}
    smoke = has_smoke(fact)
    truth = CLASS_KO.get(fact.get("class", ""), fact.get("class", ""))
    want_spread = spread_of(fact.get("area_ratio")) if smoke else "없음"
    out = {}

    out["smoke_present"] = int(bool(observed.get("smoke_present")) == smoke)

    said_color = observed.get("smoke_color")
    if not smoke:
        out["smoke_color"] = int(said_color == "없음")
    else:
        ok = COLOR_OK.get(truth)
        out["smoke_color"] = None if ok is None else int(said_color in ok)

    out["spread"] = (None if want_spread is None
                     else int(observed.get("spread") == want_spread))

    flame = fact.get("flame")
    out["flame_visible"] = (None if flame is None
                            else int(bool(observed.get("flame_visible")) == bool(flame)))
    # terrain 은 자동으로 맞출 정답이 없다. 사람이 사진을 보고 채운다.
    out["terrain"] = None
    return out


def score_one(fact, desc, observed=None):
    """한 건을 채점한다. 자동으로 못 정하는 것은 None 으로 둔다."""
    names = [u["name"] for u in fact.get("land_use", [])]
    top = names[0] if names else None
    radius = fact.get("error_radius_m")
    verdict = VERDICT_KO.get(fact.get("verdict"), fact.get("verdict"))
    truth = CLASS_KO.get(fact.get("class", ""), fact.get("class", ""))

    said_right = truth and truth in desc
    said_wrong = [c for c in CLASSES
                  if c in desc and c != truth
                  and not (any(k in c for k in SMOKE) and any(k in truth for k in SMOKE))]

    e_what = bool(said_right)
    e_where = bool(top and top in desc) or bool(re.search(r"\d{2,3}\.\d{3,}", desc))
    e_risk = any(w in desc for w in RISK_WORDS)
    e_unc = (bool(radius) and str(radius) in desc) or any(h in desc for h in HEDGE)
    elements = sum([e_what, e_where, e_risk, e_unc])

    # 판정 준수는 verify.py 의 규칙을 그대로 쓴다. 연기가 없는 사진은 알람이 뜰
    # 상황이 아니므로 위치 표현을 요구하지 않고 분모에서 뺀다.
    smoke = has_smoke(fact)
    if smoke:
        bad = verify.check(desc, alert_of(fact), observed)
        rule_ok, rule_bad = int(not bad), " / ".join(bad)
    else:
        rule_ok, rule_bad = None, ""

    # 과신은 위치가 확실하지 않은데 용도 이름을 유보 없이 말한 경우로 본다.
    # 단정 표현 목록으로 잡으면 "~이다" 가 걸려 거의 전부 과신이 된다.
    overconf = None
    if verdict in ("모호", "미확정"):
        mentioned = any(n and n in desc for n in names)
        overconf = bool(mentioned) and not any(h in desc for h in HEDGE)

    return {
        "규칙.판정준수": "" if rule_ok is None else rule_ok,
        "규칙.위반": rule_bad,
        "규칙.과신": "" if overconf is None else int(overconf),
        "규칙.무엇": int(e_what), "규칙.어디": int(e_where),
        "규칙.위험대상": int(e_risk), "규칙.불확실성": int(e_unc),
        "연기있음": int(smoke),
        "규칙.요소수": elements if smoke else "",
        "규칙.회피": int(elements <= 1) if smoke else "",
        "VLM.잘못봄": int(bool(said_wrong)),
        "VLM.잘못본단어": ",".join(said_wrong),
        "사람확인필요": int(True),      # 없는 것과 근거 없는 추론은 자동으로 못 센다
    }


def rate(rows, key):
    """비어 있지 않은 값만 모아 비율을 낸다. 분모가 0 이면 None 이다."""
    vals = [r[key] for r in rows if r.get(key) != "" and r.get(key) is not None]
    return (sum(vals) / len(vals)) if vals else None


def pct(v):
    return "-" if v is None else f"{v:.0%}"


def table(title, groups, extra_cols=()):
    """조건별 또는 방식별 집계표를 찍는다."""
    print(f"\n{title}")
    head = (f"{'':<12}{'건수':>5}{'형식':>6}{'제외':>5}{'판정준수':>9}{'과신률':>8}"
            f"{'요소포함':>9}{'회피':>5}{'잘못봄':>7}")
    for c in extra_cols:
        head += f"{c:>9}"
    print(head)
    for name in sorted(groups):
        rs = groups[name]
        ok = [r for r in rs if r["VLM.형식준수"]]
        # 요소와 회피는 연기가 있는 사진만 센다. 연기가 없으면 "연기가 보이지
        # 않는다" 가 맞는 답인데 요소가 적어 회피로 잡힌다.
        sm = [r for r in ok if r.get("연기있음") == 1]
        line = (f"{name:<12}{len(rs):>5}{len(ok) / len(rs):>6.0%}{len(rs) - len(ok):>5}"
                f"{pct(rate(ok, '규칙.판정준수')):>8}{pct(rate(ok, '규칙.과신')):>8}"
                f"{pct(rate(sm, '규칙.요소수') / 4 if rate(sm, '규칙.요소수') is not None else None):>8}"
                f"{sum(r['규칙.회피'] for r in sm):>6}"
                f"{sum(r['VLM.잘못봄'] for r in ok):>7}")
        for c in extra_cols:
            vals = [r[c] for r in rs if r.get(c) not in ("", None)]
            line += f"{(f'{sum(vals) / len(vals):.1f}' if vals else '-'):>9}"
        print(line)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit("정답 파일과 출력 파일을 적어 주세요.")
    FACTS, OUTS = sys.argv[1], sys.argv[2:]

    facts = {}
    for line in open(FACTS, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            facts[r["image"]] = r

    rows = []
    for path in OUTS:
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            o = json.loads(line)
            fact = facts.get(o["image"])
            if fact is None:
                print(f"  정답 없음: {o['image']}")
                continue
            # 실행기가 기록한 판정과 정답 파일의 판정이 다르면 알려 준다.
            # 다른 알람으로 돌린 결과를 섞어 채점하면 판정 준수율이 뜻을 잃는다.
            rec_v = o.get("verdict")
            fact_v = VERDICT_KO.get(fact.get("verdict"), fact.get("verdict"))
            if rec_v and rec_v != fact_v:
                print(f"  판정 불일치: {o['image']} 출력 {rec_v} / 정답 {fact_v}")

            observed = o.get("observed") if isinstance(o.get("observed"), dict) else None
            desc = desc_of(o)
            base = {"image": o["image"], "model": o.get("model", ""),
                    "cond": o.get("cond", ""), "방식": o.get("method", "프롬프트"),
                    "VLM.형식준수": int(desc is not None),
                    "VLM.스키마준수": ("" if "schema_ok" not in o
                                   else int(bool(o["schema_ok"]))),
                    "재생성횟수": o.get("attempts", ""),
                    "초": o.get("sec", ""),
                    "설명": desc if desc is not None else (o.get("text") or "")[:120]}
            fv = {f"VLM.{FACT_KO[k]}": ("" if v is None else v)
                  for k, v in score_facts(observed, fact).items()}
            # 형식을 안 지킨 건은 다른 지표의 분모에서 뺀다.
            scored = score_one(fact, desc, observed) if desc is not None else {}
            rows.append({**base, **fv, **scored})

    os.makedirs("results", exist_ok=True)
    cols = ["image", "model", "cond", "방식",
            "VLM.형식준수", "VLM.스키마준수", "VLM.잘못봄", "VLM.잘못본단어",
            "VLM.연기유무", "VLM.연기색", "VLM.퍼짐", "VLM.화염", "VLM.지형",
            "규칙.판정준수", "규칙.위반", "규칙.과신", "규칙.무엇", "규칙.어디",
            "규칙.위험대상", "규칙.불확실성", "규칙.요소수", "규칙.회피", "연기있음",
            "재생성횟수", "초", "사람확인필요", "설명"]
    with open("results/scores.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # 정보량 비교는 방식 (i) 로 네 조건을 본다.
    by_cond = defaultdict(list)
    for r in rows:
        if r["방식"] == "프롬프트":
            by_cond[r["cond"]].append(r)
    if by_cond:
        table("정보량 비교 (방식 i 프롬프트)", by_cond)

    # 생성 방식 비교는 조건 D 로 세 방식을 본다.
    by_method = defaultdict(list)
    for r in rows:
        if r["cond"] == "D":
            by_method[r["방식"]].append(r)
    if len(by_method) > 1:
        table("생성 방식 비교 (조건 D)", by_method, extra_cols=("재생성횟수", "초"))

    # 사실 정확도는 구조화 출력이 있는 건만 센다.
    facted = [r for r in rows if r.get("VLM.연기유무") not in ("", None)]
    if facted:
        print(f"\n사실 정확도 (구조화 출력 {len(facted)}건)")
        keys = [f"VLM.{FACT_KO[k]}" for k in FACT_FIELDS]
        for k in keys:
            print(f"  {k.split('.')[1]:<10}{pct(rate(facted, k)):>5}")
        allok = [r for r in facted
                 if all(r.get(k) == 1 for k in keys if r.get(k) not in ("", None))]
        print(f"  {'4항목모두':<10}{len(allok) / len(facted):>5.0%}")
        print("  지형은 자동 정답이 없다. scores.csv 의 VLM.지형 칸을 사람이 채운다.")

    print(f"\n저장: results/scores.csv")
    print("환각 중 '없는 것'과 '근거 없는 추론'은 자동으로 셀 수 없다.")
    print("scores.csv 의 설명 칸을 두 사람이 읽고 환각 칸을 채운다.")
    print("'잘못봄'은 자동으로 채워져 있으니 그 부분은 다시 보지 않아도 된다.")


if __name__ == "__main__":
    main()
