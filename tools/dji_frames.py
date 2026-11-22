# -*- coding: utf-8 -*-
"""DJI 드론 사진에서 위치와 카메라 각도를 꺼내 파이프라인 프레임 JSON 으로 만든다.

DJI 사진(JPEG)에는 촬영할 때의 값이 XMP(drone-dji:...)와 EXIF 에 들어 있다. 이 값을
docs/schema.md 의 camera 블록으로 옮겨, src/pipeline/run.py --frames 와 src/geo/transform.py 가 그대로
읽게 한다. 대상 기체는 Phantom 4 Pro V2.0(카메라 FC6310 계열, 1인치 센서 13.2x8.8mm, 초점거리 8.8mm)이다.

    # 1. 사진 폴더 -> 프레임 폴더 (사진을 복사하고 사진마다 JSON 을 만든다)
    python tools/dji_frames.py frames <사진 폴더> --out data/drone_1120

    # 2. 지도에서 좌표를 아는 점(표적)을 사진에서 짚어 위치 오차를 잰다
    python tools/dji_frames.py point data/drone_1120/DJI_0012.json --px 2731 1980 --truth 37.12345 127.12345
    python tools/dji_frames.py points 표적.csv --frames data/drone_1120 --out results/drone_error.csv

camera 블록으로 옮기는 값
    lat, lon       drone-dji:GpsLatitude, GpsLongitude (없으면 EXIF GPS)
    alt_m, alt_ref drone-dji:RelativeAltitude, "지면". 이륙 지점 기준 높이다. 표적의 땅 높이가 이륙 지점과
                   다르면 그만큼 틀린다. --alt-mode absolute 면 AbsoluteAltitude 를 "해발"로 넣는다(DEM 필요)
    heading_deg    drone-dji:GimbalYawDegree (카메라가 향한 방향. 기체 방향 FlightYawDegree 가 아니다)
    pitch_deg      drone-dji:GimbalPitchDegree (-90 이 수직 하향)
    hfov_deg       센서 폭과 초점거리로 계산. 사진 비율에 따라 센서를 쓰는 폭이 달라진다
                   (3:2 는 73.7도, 4:3 은 67.4도)

points 의 CSV 열: photo, px, py, truth_lat, truth_lon, name(선택)
    photo 는 프레임 JSON 이름(확장자 빼고), px, py 는 사진 원본 크기의 픽셀이다.
"""
import argparse
import csv
import json
import math
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "geo"))

SENSOR_W_MM = 13.2           # Phantom 4 Pro V2 1인치 센서
FULL_W_PX = 5472             # 3:2 일 때 가로 픽셀. 4:3(4864)은 센서 가로를 덜 쓴다
FOCAL_MM = 8.8
NEED = ["lat", "lon", "alt_m", "heading_deg", "pitch_deg"]


def read_xmp(path):
    data = Path(path).read_bytes()
    a = data.find(b"<x:xmpmeta")
    b = data.find(b"</x:xmpmeta>")
    if a < 0 or b < 0:
        return {}
    x = data[a:b].decode("utf-8", "ignore")
    out = dict(re.findall(r'drone-dji:(\w+)="([^"]*)"', x))
    out.update(dict(re.findall(r"<drone-dji:(\w+)>([^<]*)</drone-dji:\1>", x)))
    return out


def read_exif(path):
    from PIL import Image
    from PIL.ExifTags import TAGS, GPSTAGS
    img = Image.open(path)
    ex = img.getexif()
    tags = {TAGS.get(k, k): v for k, v in ex.items()}
    try:
        sub = ex.get_ifd(0x8769)
        tags.update({TAGS.get(k, k): v for k, v in sub.items()})
    except Exception:
        pass
    gps = {}
    try:
        gps = {GPSTAGS.get(k, k): v for k, v in ex.get_ifd(0x8825).items()}
    except Exception:
        pass
    return img.size, tags, gps


def dms(v, ref):
    d, m, s = (float(x) for x in v)
    x = d + m / 60 + s / 3600
    return -x if ref in ("S", "W") else x


def num(x):
    try:
        return float(str(x).strip().replace("+", ""))
    except (TypeError, ValueError):
        return None


def hfov_for(width_px, focal_mm=FOCAL_MM):
    sensor_w = SENSOR_W_MM * min(1.0, width_px / FULL_W_PX)
    return math.degrees(2 * math.atan(sensor_w / (2 * focal_mm)))


def frame_of(path, index, alt_mode="relative", video_id="DRONE"):
    (w, h), tags, gps = read_exif(path)
    x = read_xmp(path)
    lat = num(x.get("GpsLatitude"))
    lon = num(x.get("GpsLongitude") or x.get("GpsLongtitude"))      # 옛 펌웨어는 철자가 틀리게 저장한다
    if (lat is None or lon is None) and gps.get("GPSLatitude"):
        lat = dms(gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N"))
        lon = dms(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E"))
    if alt_mode == "absolute":
        alt, ref = num(x.get("AbsoluteAltitude")), "해발"
    else:
        alt, ref = num(x.get("RelativeAltitude")), "지면"
    focal = num(tags.get("FocalLength")) or FOCAL_MM
    cam = {"lat": lat, "lon": lon, "alt_m": alt, "alt_ref": ref,
           "heading_deg": num(x.get("GimbalYawDegree")), "pitch_deg": num(x.get("GimbalPitchDegree")),
           "hfov_deg": round(hfov_for(w, focal), 2), "source": "실제"}
    if cam["heading_deg"] is not None:
        cam["heading_deg"] = cam["heading_deg"] % 360
    raw = {k: x.get(k) for k in ("GpsLatitude", "GpsLongitude", "GpsLongtitude", "AbsoluteAltitude", "RelativeAltitude",
                                 "GimbalRollDegree", "GimbalYawDegree", "GimbalPitchDegree",
                                 "FlightRollDegree", "FlightYawDegree", "FlightPitchDegree") if x.get(k) is not None}
    raw.update({"Model": tags.get("Model"), "FocalLength": focal, "DateTimeOriginal": tags.get("DateTimeOriginal")})
    t = tags.get("DateTimeOriginal")
    t = t.replace(":", "-", 2).replace(" ", "T") if isinstance(t, str) else None
    return {"schema_version": "0.3", "video_id": video_id, "frame_id": index, "time": t,
            "image_size": {"width": w, "height": h}, "camera": cam, "detections": [],
            "image": Path(path).name, "dji": raw}


def checks(fr):
    """찍은 값 중 위치 계산에 걸릴 것을 짧게 적는다."""
    c, r, out = fr["camera"], fr["dji"], []
    miss = [k for k in NEED if c.get(k) is None]
    if miss:
        out.append("없음: " + ", ".join(miss))
    roll = num(r.get("GimbalRollDegree"))
    if roll is not None and abs(roll) > 2:
        out.append(f"짐벌 롤 {roll:g}도 (계산은 롤 0 가정)")
    gy, fy = num(r.get("GimbalYawDegree")), num(r.get("FlightYawDegree"))
    if gy is not None and fy is not None:
        d = abs((gy - fy + 180) % 360 - 180)
        if d > 5:
            out.append(f"짐벌과 기체 방향이 {d:.0f}도 다름 (짐벌 값을 씀)")
    if c.get("pitch_deg") is not None and c["pitch_deg"] > -5:
        out.append(f"짐벌 각도 {c['pitch_deg']:g}도로 거의 수평 (지면 좌표를 못 구할 수 있음)")
    if c.get("alt_m") is not None and c["alt_m"] <= 0:
        out.append(f"높이 {c['alt_m']:g}m (이륙 전 사진?)")
    return out


def cmd_frames(a):
    src = Path(a.photos)
    photos = sorted([p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg")])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, p in enumerate(photos):
        fr = frame_of(p, i, a.alt_mode, a.video_id or src.name)
        (out / (p.stem + ".json")).write_text(json.dumps(fr, ensure_ascii=False, indent=1), encoding="utf-8")
        if not a.no_copy and not (out / p.name).exists():
            shutil.copy2(p, out / p.name)
        c = fr["camera"]
        warn = checks(fr)
        rows.append({"photo": p.stem, "time": fr["time"], "lat": c["lat"], "lon": c["lon"], "alt_m": c["alt_m"],
                     "heading_deg": c["heading_deg"], "pitch_deg": c["pitch_deg"], "hfov_deg": c["hfov_deg"],
                     "width": fr["image_size"]["width"], "height": fr["image_size"]["height"],
                     "model": fr["dji"].get("Model"), "check": "; ".join(warn)})
        print(f"{p.name}: {c['lat']}, {c['lon']} 높이 {c['alt_m']} 방향 {c['heading_deg']} 각도 {c['pitch_deg']} "
              f"화각 {c['hfov_deg']}" + (f"  !! {'; '.join(warn)}" if warn else ""))
    with open(out / "frames_summary.csv", "w", encoding="utf-8-sig", newline="") as fo:
        if rows:
            w = csv.DictWriter(fo, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    bad = sum(1 for r in rows if r["check"].startswith("없음"))
    print(f"\n사진 {len(rows)}장, 위치 계산에 필요한 값이 빠진 사진 {bad}장 -> {out}")


def dem_of(a):
    if not a.dem:
        return None
    from dem import Dem
    return Dem(a.dem)


def measure(frame, px, py, tlat, tlon, dem=None):
    from transform import to_position
    det = {"polygon": None, "bbox": [px - 0.5, py - 1, 1, 1]}      # 대표 픽셀이 (px, py) 가 되게
    pos, info = to_position(frame, det, dem=dem)
    R = 6378137.0
    dy = math.radians(tlat - pos["lat"]) * R
    dx = math.radians(tlon - pos["lon"]) * R * math.cos(math.radians(tlat))
    err = math.hypot(dx, dy)
    cam = frame["camera"]
    dlat = math.radians(tlat - cam["lat"]) * R
    dlon = math.radians(tlon - cam["lon"]) * R * math.cos(math.radians(tlat))
    return {"est_lat": pos["lat"], "est_lon": pos["lon"], "err_m": round(err, 1),
            "radius_m": pos["error_radius_m"], "inside": err <= pos["error_radius_m"],
            "dist_m": round(math.hypot(dlat, dlon), 1), "depression_deg": info["depression_deg"],
            "shallow": info["shallow"], "err_east_m": round(dx, 1), "err_north_m": round(dy, 1)}


def cmd_point(a):
    fr = json.loads(Path(a.frame).read_text(encoding="utf-8"))
    m = measure(fr, a.px[0], a.px[1], a.truth[0], a.truth[1], dem_of(a))
    print(f"추정 {m['est_lat']}, {m['est_lon']}  오차 {m['err_m']}m (동 {m['err_east_m']}, 북 {m['err_north_m']})  "
          f"반경 {m['radius_m']}m 안 {'예' if m['inside'] else '아니오'}  드론과 표적 거리 {m['dist_m']}m  "
          f"내려다보는 각 {m['depression_deg']}도")


def cmd_points(a):
    dem = dem_of(a)
    rows = []
    for r in csv.DictReader(open(a.csv, encoding="utf-8-sig")):
        fr = json.loads((Path(a.frames) / (r["photo"] + ".json")).read_text(encoding="utf-8"))
        m = measure(fr, float(r["px"]), float(r["py"]), float(r["truth_lat"]), float(r["truth_lon"]), dem)
        rows.append({**r, **m, "pitch_deg": fr["camera"]["pitch_deg"], "alt_m": fr["camera"]["alt_m"]})
    if not rows:
        print("표적이 없다")
        return
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8-sig", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    errs = sorted(r["err_m"] for r in rows)
    print(f"표적 {len(rows)}개: 오차 중앙값 {errs[len(errs) // 2]}m, 최대 {errs[-1]}m, "
          f"반경 안 {sum(r['inside'] for r in rows)}/{len(rows)}")
    me = sum(r["err_east_m"] for r in rows) / len(rows)
    mn = sum(r["err_north_m"] for r in rows) / len(rows)
    print(f"오차 평균 방향: 동 {me:.1f}m, 북 {mn:.1f}m (한쪽으로 쏠리면 방향이나 고도 값이 치우친 것)")
    print(f"-> {a.out}")


def main():
    ap = argparse.ArgumentParser(description="DJI 사진 -> 프레임 JSON, 표적 위치 오차")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("frames")
    f.add_argument("photos")
    f.add_argument("--out", required=True)
    f.add_argument("--alt-mode", choices=["relative", "absolute"], default="relative")
    f.add_argument("--video-id")
    f.add_argument("--no-copy", action="store_true", help="사진을 프레임 폴더로 복사하지 않는다")
    p = sub.add_parser("point")
    p.add_argument("frame")
    p.add_argument("--px", type=float, nargs=2, required=True)
    p.add_argument("--truth", type=float, nargs=2, required=True, metavar=("LAT", "LON"))
    p.add_argument("--dem", nargs="*")
    q = sub.add_parser("points")
    q.add_argument("csv")
    q.add_argument("--frames", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--dem", nargs="*")
    a = ap.parse_args()
    {"frames": cmd_frames, "point": cmd_point, "points": cmd_points}[a.cmd](a)


if __name__ == "__main__":
    main()
