"""입출력 JSON 형식 검사기 (docs/schema.md v0.3).

탐지 결과와 알람 JSON이 정해진 키, 타입, 값 목록을 지키는지 검사한다.
파일 종류는 키로 자동 구분한다 (`detections`가 있으면 탐지, `alert_id`가 있으면 알람).

사용 예:
    python tools/check_schema.py schema/*.example.json
    python tools/check_schema.py results/pred/*.json

문제가 없으면 OK, 있으면 파일별로 어떤 필드가 왜 틀렸는지 출력한다.
하나라도 틀리면 종료 코드 1을 돌려준다.
"""
import glob
import json
import sys

VERSION = "0.3"
NUM = (int, float)

# 탐지 결과: (경로, 타입, 허용 값 목록 또는 None)
DET_TOP = [
    ("schema_version", str, [VERSION]),
    ("video_id", str, None),
    ("frame_id", int, None),
    ("time", str, None),
    ("image", str, None),
    ("image_size", dict, None),
    ("detections", list, None),
    ("camera", dict, None),
    ("latency_ms", dict, None),
]
DET_ITEM = [
    ("id", str, None),
    ("class", str, ["연기"]),
    ("score", NUM, None),
    ("bbox", list, None),
    ("polygon", (list, type(None)), None),
    ("area_px", int, None),
    ("area_ratio", NUM, None),
]
CAMERA = [
    ("lat", NUM, None),
    ("lon", NUM, None),
    ("alt_m", NUM, None),
    ("alt_ref", str, ["지면", "해발"]),
    ("heading_deg", NUM, None),
    ("pitch_deg", NUM, None),
    ("hfov_deg", NUM, None),
    ("source", str, ["실제", "가상"]),
]
ALERT_TOP = [
    ("schema_version", str, [VERSION]),
    ("alert_id", str, None),
    ("time", str, None),
    ("detection_id", str, None),
    ("position", dict, None),
    ("location_verdict", str, ["확정", "모호", "미확정"]),
    ("land_use", list, None),
    ("verdict_rule", dict, None),
    ("evidence", (dict, type(None)), None),
    ("observed", (dict, type(None)), None),
    ("description", (str, type(None)), None),
    ("description_method", (str, type(None)), ["프롬프트", "검증재생성", "구조화템플릿", None]),
    ("latency_ms", dict, None),
]
OBSERVED = [
    ("smoke_present", bool, None),
    ("smoke_color", str, ["검은색", "회색", "흰색", "없음"]),
    ("spread", str, ["좁음", "보통", "넓음", "없음"]),
    ("flame_visible", bool, None),
    ("terrain", str, ["산림", "농경지", "주거지", "도로", "물", "기타"]),
]
LATENCY = ["detect", "geo_transform", "geo_query", "verdict", "vlm",
           "detect_to_alert", "alert_to_description"]

paths = []
for a in sys.argv[1:]:
    paths += sorted(glob.glob(a)) or [a]
if not paths:
    sys.exit("사용법: python tools/check_schema.py <JSON 파일들>")

bad_files = 0
for path in paths:
    errs = []
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        print(f"[FAIL] {path}\n  - JSON 읽기 실패: {e}")
        bad_files += 1
        continue

    # 검사할 (접두어, 객체, 규칙) 묶음을 쌓아 두고 한 번에 돈다
    if "detections" in d:
        kind = "탐지"
        jobs = [("", d, DET_TOP)]
        if isinstance(d.get("camera"), dict):
            jobs.append(("camera.", d["camera"], CAMERA))
        for i, det in enumerate(d.get("detections") or []):
            jobs.append((f"detections[{i}].", det, DET_ITEM))
    elif "alert_id" in d:
        kind = "알람"
        jobs = [("", d, ALERT_TOP)]
        if isinstance(d.get("observed"), dict):
            jobs.append(("observed.", d["observed"], OBSERVED))
    else:
        print(f"[FAIL] {path}\n  - 탐지(detections)인지 알람(alert_id)인지 알 수 없다")
        bad_files += 1
        continue

    for prefix, obj, rules in jobs:
        if not isinstance(obj, dict):
            errs.append(f"{prefix} 객체가 아니다")
            continue
        names = {r[0] for r in rules}
        for k in obj:
            if k not in names:
                errs.append(f"{prefix}{k}: 정의되지 않은 키")
        for key, typ, allowed in rules:
            if key not in obj:
                errs.append(f"{prefix}{key}: 키 없음")
                continue
            v = obj[key]
            if isinstance(v, bool) and typ is not bool and bool not in (typ if isinstance(typ, tuple) else (typ,)):
                errs.append(f"{prefix}{key}: 타입이 bool")
            elif not isinstance(v, typ):
                errs.append(f"{prefix}{key}: 타입이 {type(v).__name__}")
            elif allowed is not None and v not in allowed:
                errs.append(f"{prefix}{key}: 허용 값이 아님 ({v!r}, 허용: {allowed})")

    # 형식 말고 값 사이의 관계 검사
    if kind == "탐지":
        size = d.get("image_size") or {}
        if set(size) != {"width", "height"}:
            errs.append("image_size: width, height 둘만 있어야 한다")
        if ".000" not in d.get("time", "") and "." not in d.get("time", "")[-12:]:
            errs.append("time: 밀리초까지 써야 한다")
        for i, det in enumerate(d.get("detections") or []):
            b = det.get("bbox")
            if not (isinstance(b, list) and len(b) == 4):
                errs.append(f"detections[{i}].bbox: [x, y, 너비, 높이] 4개여야 한다")
            elif size and (b[0] + b[2] > size.get("width", 1e9) or b[1] + b[3] > size.get("height", 1e9)):
                errs.append(f"detections[{i}].bbox: 이미지 밖으로 나감")
            pg = det.get("polygon")
            if isinstance(b, list) and len(b) == 4 and isinstance(pg, list) and pg:
                xs = [p[0] for p in pg]
                ys = [p[1] for p in pg]
                tight = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
                if max(abs(u - w) for u, w in zip(b, tight)) > 2:
                    errs.append(f"detections[{i}].bbox: 폴리곤을 감싸는 박스 {tight}와 다름 "
                                f"([x1, y1, x2, y2]로 낸 건 아닌지 확인)")
            s = det.get("score")
            if isinstance(s, NUM) and not 0 <= s <= 1:
                errs.append(f"detections[{i}].score: 0~1 범위 밖")
            if size and isinstance(det.get("area_px"), int) and isinstance(det.get("area_ratio"), NUM):
                expect = det["area_px"] / (size["width"] * size["height"])
                if abs(expect - det["area_ratio"]) > 0.001:
                    errs.append(f"detections[{i}].area_ratio: area_px/이미지 넓이({expect:.4f})와 다름")
            if not det.get("id", "").startswith(f"{d.get('video_id')}-{d.get('frame_id')}-"):
                errs.append(f"detections[{i}].id: '{{video_id}}-{{frame_id}}-{{순번}}' 형식이 아님")
    else:
        lu = d.get("land_use") or []
        ratios = [x.get("ratio", 0) for x in lu if isinstance(x, dict)]
        if lu and abs(sum(ratios) - 1) > 0.01:
            errs.append(f"land_use: 비율 합이 {sum(ratios):.2f} (1이어야 한다)")
        if ratios != sorted(ratios, reverse=True):
            errs.append("land_use: 비율 내림차순이 아님")
        if not lu and d.get("location_verdict") != "미확정":
            errs.append("land_use가 비었는데 판정이 미확정이 아님")
        if len(lu) == 1 and d.get("location_verdict") == "모호":
            errs.append("land_use가 1개인데 판정이 모호")
        ob = d.get("observed")
        if isinstance(ob, dict) and ob.get("smoke_present") is False:
            if ob.get("smoke_color") != "없음" or ob.get("spread") != "없음":
                errs.append("observed: 연기가 없으면 smoke_color, spread는 '없음'")
        if d.get("description_method") == "구조화템플릿" and not isinstance(ob, dict):
            errs.append("구조화템플릿인데 observed가 없음")
        if d.get("description") is not None and d.get("description_method") is None:
            errs.append("description이 있는데 description_method가 null")
        lat = d.get("latency_ms") or {}
        for k in LATENCY:
            if k not in lat:
                errs.append(f"latency_ms.{k}: 키 없음 (아직 안 끝났으면 null)")

    if errs:
        bad_files += 1
        print(f"[FAIL] {path} ({kind})")
        for e in errs:
            print(f"  - {e}")
    else:
        print(f"[OK]   {path} ({kind})")

print(f"\n{len(paths) - bad_files}/{len(paths)}개 통과")
sys.exit(1 if bad_files else 0)
