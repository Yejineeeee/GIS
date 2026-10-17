# -*- coding: utf-8 -*-
"""탐지 픽셀 위치를 실제 좌표와 오차 반경으로 바꾼다 (WBS 3.3, 초안).

입력은 탐지 결과 JSON(docs/schema.md v0.3)이고, 출력은 알람의 `position`
(lat, lon, error_radius_m, virtual)이다.

    python3 src/geo/transform.py 탐지결과.json [--alt 120] [--max-tilt 45]

계산 방법
    1. 탐지 영역의 대표 픽셀을 고른다 (폴리곤 무게중심, 없으면 bbox 중심).
    2. 핀홀 카메라로 그 픽셀 방향의 광선을 만들고, 평평한 지면(고도 alt_m 아래)과
       만나는 점을 구한다. 드론 위치에서 동쪽, 북쪽으로 몇 m인지 나온다.
    3. 그 거리만큼 드론 위도·경도를 옮긴다.
    4. 오차 반경 = GNSS 위치 오차 + (각도를 모를 때) 고도 x tan(최대 기울기)
       + (기수 방향을 모를 때) 화면 중심에서 떨어진 거리.

데이터에 없는 값은 아래 가정값을 쓰고, 어떤 가정을 썼는지 함께 돌려준다.
지형 높낮이는 반영하지 않는다(평지 가정). 외부 패키지 없이 동작한다.
"""
import argparse
import json
import math
import sys

# ---------- 가정값 (팀에서 확정 전) ----------
ALT_DEFAULT_M = 120.0      # 비행 높이(지면 기준). 팀 결정(10/17), 예진 파일럿 코드와 같음
GNSS_ERROR_M = 5.0         # 드론 GNSS 위치 오차. 제안서 '일반 GNSS 기준 약 ±5m'
MAX_TILT_DEG = 30.0        # 촬영 각도를 모를 때 가정하는 최대 기울기. 팀 결정(10/17)
                           # 120m x tan30 + 5m = 약 74m로 3.5 실험 최대 반경(100m) 안에 들어옴
HFOV_DEFAULT_DEG = math.degrees(2 * math.atan(13.2 / (2 * 8.8)))  # gsd.md 예시 카메라, 약 73.7도
EARTH_R = 6378137.0


def target_pixel(det):
    """탐지 하나의 대표 픽셀 (x, y)."""
    poly = det.get("polygon")
    if poly and len(poly) >= 3:
        a = cx = cy = 0.0
        for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
            c = x1 * y2 - x2 * y1
            a += c
            cx += (x1 + x2) * c
            cy += (y1 + y2) * c
        if abs(a) > 1e-9:
            return cx / (3 * a), cy / (3 * a)
    x, y, w, h = det["bbox"]
    return x + w / 2, y + h / 2


def ground_offset(px, py, width, height, alt_m, heading_deg, pitch_deg, hfov_deg):
    """픽셀 (px, py)가 가리키는 지면 점이 드론에서 동쪽(e), 북쪽(n)으로 몇 m인지.

    pitch_deg: -90이 수직 하향, 0이 수평. heading_deg: 진북 0도, 시계 방향.
    광선이 지면에 닿지 않으면(수평선 위) None.
    """
    f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)   # 픽셀 단위 초점거리
    dx = px - width / 2        # 오른쪽 +
    dy = py - height / 2       # 아래쪽 +
    p, h = math.radians(pitch_deg), math.radians(heading_deg)
    # 수평면 기준 축: 앞(F), 오른쪽(R), 위(U)
    # 카메라 광축 = cos(p)F + sin(p)U, 화면 위쪽 = -sin(p)F + cos(p)U, 화면 오른쪽 = R
    ray_f = f * math.cos(p) + dy * math.sin(p)
    ray_u = f * math.sin(p) - dy * math.cos(p)
    ray_r = dx
    if ray_u >= -1e-9:
        return None
    t = alt_m / -ray_u
    fwd, right = ray_f * t, ray_r * t
    e = fwd * math.sin(h) + right * math.cos(h)
    n = fwd * math.cos(h) - right * math.sin(h)
    return e, n


def move(lat, lon, e, n):
    """위도·경도를 동쪽 e m, 북쪽 n m 옮긴다 (수 km 이내 근사)."""
    dlat = math.degrees(n / EARTH_R)
    dlon = math.degrees(e / (EARTH_R * math.cos(math.radians(lat))))
    return lat + dlat, lon + dlon


def to_position(frame, det, alt_default=ALT_DEFAULT_M, max_tilt=MAX_TILT_DEG):
    """탐지 하나를 알람 position으로 바꾼다. (position, 사용한 가정 목록)을 돌려준다."""
    cam = frame.get("camera") or {}
    size = frame["image_size"]
    w, h = size["width"], size["height"]
    notes = []

    if cam.get("lat") is None or cam.get("lon") is None:
        raise ValueError("camera.lat, camera.lon이 비어 있음 (가상 배치 위치를 먼저 넣어야 함)")

    alt = cam.get("alt_m")
    if alt is None:
        alt = alt_default
        notes.append(f"비행 높이 가정 {alt:g}m")
    elif cam.get("alt_ref") not in ("지면", "AGL"):
        notes.append(f"alt_ref={cam.get('alt_ref')}: 지면 기준이 아니라 가정값 {alt_default:g}m 사용")
        alt = alt_default

    hfov = cam.get("hfov_deg")
    if hfov is None:
        hfov = HFOV_DEFAULT_DEG
        notes.append(f"화각 가정 {hfov:.1f}도 (gsd.md 예시 카메라)")

    pitch = cam.get("pitch_deg")
    pitch_known = pitch is not None
    if not pitch_known:
        pitch = -90.0
        notes.append(f"촬영 각도 모름: 수직 하향 가정, 최대 기울기 {max_tilt:g}도만큼 반경 추가")

    heading = cam.get("heading_deg")
    heading_known = heading is not None
    if not heading_known:
        heading = 0.0
        notes.append("기수 방향 모름: 드론 위치를 중심으로 두고 화면 중심과의 거리만큼 반경 추가")

    px, py = target_pixel(det)
    off = ground_offset(px, py, w, h, alt, heading, pitch, hfov)
    if off is None:
        raise ValueError("탐지 위치가 수평선 위라 지면 좌표를 구할 수 없음")
    e, n = off
    dist = math.hypot(e, n)

    radius = GNSS_ERROR_M
    if not pitch_known:
        radius += alt * math.tan(math.radians(max_tilt))
    if heading_known:
        lat, lon = move(cam["lat"], cam["lon"], e, n)
    else:
        lat, lon = cam["lat"], cam["lon"]
        radius += dist

    position = {
        "lat": round(lat, 7),
        "lon": round(lon, 7),
        "error_radius_m": round(radius, 1),
        "virtual": cam.get("source") in ("가상", "virtual"),
    }
    info = {"pixel": [round(px, 1), round(py, 1)], "ground_offset_m": round(dist, 1),
            "alt_m": alt, "assumptions": notes}
    return position, info


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("detection_json")
    ap.add_argument("--alt", type=float, default=ALT_DEFAULT_M, help="비행 높이 가정값(m)")
    ap.add_argument("--max-tilt", type=float, default=MAX_TILT_DEG, help="각도를 모를 때 최대 기울기(도)")
    a = ap.parse_args()

    with open(a.detection_json, encoding="utf-8") as f:
        frame = json.load(f)
    if not frame.get("detections"):
        print("탐지 없음")
        return
    for det in frame["detections"]:
        try:
            pos, info = to_position(frame, det, a.alt, a.max_tilt)
        except ValueError as e:
            print(f"{det.get('id')}: 변환 실패 - {e}")
            continue
        print(f"{det.get('id')}: {json.dumps(pos, ensure_ascii=False)}")
        print(f"   픽셀 {info['pixel']}, 화면 중심에서 지면 {info['ground_offset_m']}m")
        for note in info["assumptions"]:
            print(f"   가정: {note}")


if __name__ == "__main__":
    sys.exit(main())
