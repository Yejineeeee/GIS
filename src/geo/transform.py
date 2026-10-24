# -*- coding: utf-8 -*-
"""탐지 픽셀 위치를 실제 좌표와 오차 반경으로 바꾼다 (WBS 3.3).

입력은 탐지 결과 JSON(docs/schema.md v0.3)이고, 출력은 알람의 `position`
(lat, lon, error_radius_m, virtual)이다.

    python3 src/geo/transform.py 탐지결과.json [--dem 37808.img 37804.img] [--geoid 25]

계산 방법
    1. 대표 픽셀: 연기 폴리곤의 맨 아래(연기 밑동). 연기는 지면에서 피어오르므로
       밑동이 발화점에 가장 가깝다. 폴리곤이 없으면 bbox 아래 가운데.
    2. 고도: 지면 기준 높이로 바꾼다. alt_ref가 해발이면 드론 아래 DEM 높이를 빼고,
       타원체면 지오이드 높이(--geoid)도 함께 뺀다.
    3. 핀홀 카메라로 그 픽셀 방향의 광선을 만들어 지면과 만나는 점을 구한다.
       DEM이 있으면 만나는 점의 지면 높이를 반영해 몇 번 다시 계산한다.
    4. 오차 반경 = GNSS 위치 오차
                 + (각도를 모를 때) 고도 x tan(최대 기울기)
                 + (기수 방향을 모를 때) 화면 중심에서 떨어진 거리
                 + 연기 밑동 높이 / tan(내려다보는 각)   (밑동이 지면보다 떠 있을 수 있음)
    5. 내려다보는 각이 얕으면(기본 15도 미만) info.shallow를 true로 둔다.
       이때는 위치를 믿기 어려우므로 판정에서 미확정으로 두는 것을 권한다.

데이터에 없는 값은 아래 가정값을 쓰고, 어떤 가정을 썼는지 함께 돌려준다.
외부 패키지 없이 동작한다.

변경 (8주차): 시뮬레이터로 찾은 두 오차를 고침
    - 고도 기준: 해발·타원체 고도를 지면 기준으로 바꿈 (전에는 가정값 120m로 넘어감)
    - 연기 높이: 대표 픽셀을 무게중심에서 맨 아래로 바꾸고, 밑동 높이 항과 얕은 각도 표시를 추가
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------- 가정값 ----------
ALT_DEFAULT_M = 120.0      # 비행 높이(지면 기준). 팀 결정(10/17)
GNSS_ERROR_M = 5.0         # 드론 GNSS 위치 오차. 제안서 '일반 GNSS 기준 약 ±5m'
MAX_TILT_DEG = 30.0        # 촬영 각도를 모를 때 가정하는 최대 기울기. 팀 결정(10/17)
HFOV_DEFAULT_DEG = math.degrees(2 * math.atan(13.2 / (2 * 8.8)))  # gsd.md 예시 카메라, 약 73.7도
SMOKE_BASE_M = 5.0         # 폴리곤 맨 아래가 지면보다 떠 있을 수 있는 높이(가정)
MIN_DEPRESSION_DEG = 15.0  # 이보다 얕게 내려다보면 위치를 믿기 어렵다고 표시
BOTTOM_BAND = 0.05         # 폴리곤 높이의 아래 5% 안 점들의 가운데를 밑동으로 씀
EARTH_R = 6378137.0


def target_pixel(det):
    """탐지 하나의 대표 픽셀 (x, y) = 연기 밑동."""
    poly = det.get("polygon")
    if poly and len(poly) >= 3:
        ys = [p[1] for p in poly]
        top, bottom = min(ys), max(ys)
        band = max((bottom - top) * BOTTOM_BAND, 1.0)
        xs = [p[0] for p in poly if p[1] >= bottom - band]
        return sum(xs) / len(xs), bottom
    x, y, w, h = det["bbox"]
    return x + w / 2, y + h


def ray(px, py, width, height, heading_deg, pitch_deg, hfov_deg):
    """픽셀 방향 광선을 (동, 북, 위) 성분으로 돌려준다."""
    f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)   # 픽셀 단위 초점거리
    dx = px - width / 2        # 오른쪽 +
    dy = py - height / 2       # 아래쪽 +
    p, h = math.radians(pitch_deg), math.radians(heading_deg)
    ray_f = f * math.cos(p) + dy * math.sin(p)
    ray_u = f * math.sin(p) - dy * math.cos(p)
    ray_r = dx
    e = ray_f * math.sin(h) + ray_r * math.cos(h)
    n = ray_f * math.cos(h) - ray_r * math.sin(h)
    return e, n, ray_u


def ground_offset(px, py, width, height, alt_m, heading_deg, pitch_deg, hfov_deg):
    """평평한 지면(드론 alt_m 아래)과 만나는 점. 드론에서 (동, 북) m. 수평선 위면 None."""
    e, n, u = ray(px, py, width, height, heading_deg, pitch_deg, hfov_deg)
    if u >= -1e-9:
        return None
    t = alt_m / -u
    return e * t, n * t


def depression_deg(px, py, width, height, pitch_deg, hfov_deg):
    """광선이 수평선 아래로 몇 도 내려가는지."""
    e, n, u = ray(px, py, width, height, 0.0, pitch_deg, hfov_deg)
    return math.degrees(math.atan2(-u, math.hypot(e, n)))


def move(lat, lon, e, n):
    """위도·경도를 동쪽 e m, 북쪽 n m 옮긴다 (수 km 이내 근사)."""
    dlat = math.degrees(n / EARTH_R)
    dlon = math.degrees(e / (EARTH_R * math.cos(math.radians(lat))))
    return lat + dlat, lon + dlon


def _ground(dem, lat, lon):
    if dem is None:
        return None
    from query import to_5179
    x, y = to_5179(lon, lat)
    return dem.height(x, y)


def agl_altitude(cam, dem, geoid_m, alt_default, notes):
    """camera 블록의 고도를 지면 기준 높이로 바꾼다."""
    alt = cam.get("alt_m")
    ref = cam.get("alt_ref")
    if alt is None:
        notes.append(f"비행 높이 가정 {alt_default:g}m")
        return alt_default, None
    if ref in ("지면", "AGL"):
        return alt, None
    if ref in ("해발", "MSL", "타원체", "ellipsoid"):
        g = _ground(dem, cam["lat"], cam["lon"])
        if g is None:
            notes.append(f"alt_ref={ref}인데 드론 아래 DEM 높이를 몰라 가정값 {alt_default:g}m 사용")
            return alt_default, None
        n = 0.0
        if ref in ("타원체", "ellipsoid"):
            if geoid_m is None:
                notes.append("타원체 고도인데 지오이드 높이(--geoid)가 없어 0으로 둠. 수십 m 틀릴 수 있음")
            else:
                n = geoid_m
        agl = alt - n - g
        notes.append(f"고도 {alt:g}m({ref}) - 지오이드 {n:g}m - 드론 아래 지면 {g:.1f}m = 지면 위 {agl:.1f}m")
        if agl <= 0:
            raise ValueError(f"지면 기준 고도가 {agl:.1f}m로 0 이하. alt_ref나 지오이드 값을 확인해야 함")
        return agl, g
    notes.append(f"alt_ref={ref}: 알 수 없는 기준이라 가정값 {alt_default:g}m 사용")
    return alt_default, None


def to_position(frame, det, alt_default=ALT_DEFAULT_M, max_tilt=MAX_TILT_DEG,
                dem=None, geoid_m=None):
    """탐지 하나를 알람 position으로 바꾼다. (position, info)를 돌려준다."""
    cam = frame.get("camera") or {}
    size = frame["image_size"]
    w, h = size["width"], size["height"]
    notes = []

    if cam.get("lat") is None or cam.get("lon") is None:
        raise ValueError("camera.lat, camera.lon이 비어 있음 (가상 배치 위치를 먼저 넣어야 함)")

    alt, ground_drone = agl_altitude(cam, dem, geoid_m, alt_default, notes)

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

    # DEM이 있고 방향을 알면, 만나는 점의 지면 높이를 반영해 다시 계산한다
    if dem is not None and heading_known:
        if ground_drone is None:
            ground_drone = _ground(dem, cam["lat"], cam["lon"])
        if ground_drone is not None:
            for _ in range(5):
                g = _ground(dem, *move(cam["lat"], cam["lon"], e, n))
                if g is None:
                    break
                eff = alt + ground_drone - g
                if eff <= 0:
                    break
                e2, n2 = ground_offset(px, py, w, h, eff, heading, pitch, hfov)
                done = math.hypot(e2 - e, n2 - n) < 0.5
                e, n = e2, n2
                if done:
                    break
            notes.append("DEM 지형 높이 반영")

    dist = math.hypot(e, n)
    dep = depression_deg(px, py, w, h, pitch, hfov)
    shallow = dep < MIN_DEPRESSION_DEG

    radius = GNSS_ERROR_M
    if not pitch_known:
        radius += alt * math.tan(math.radians(max_tilt))
    radius += SMOKE_BASE_M / math.tan(math.radians(max(dep, 1.0)))
    if heading_known:
        lat, lon = move(cam["lat"], cam["lon"], e, n)
    else:
        lat, lon = cam["lat"], cam["lon"]
        radius += dist
    if shallow:
        notes.append(f"내려다보는 각 {dep:.1f}도로 얕음: 위치를 믿기 어려워 미확정 권장")

    position = {
        "lat": round(lat, 7),
        "lon": round(lon, 7),
        "error_radius_m": round(radius, 1),
        "virtual": cam.get("source") in ("가상", "virtual"),
    }
    info = {"pixel": [round(px, 1), round(py, 1)], "ground_offset_m": round(dist, 1),
            "alt_agl_m": round(alt, 1), "depression_deg": round(dep, 1),
            "shallow": shallow, "assumptions": notes}
    return position, info


def main():
    ap = argparse.ArgumentParser(description="탐지 픽셀 → 실제 좌표와 오차 반경 (WBS 3.3)")
    ap.add_argument("detection_json")
    ap.add_argument("--alt", type=float, default=ALT_DEFAULT_M, help="비행 높이 가정값(m, 지면 기준)")
    ap.add_argument("--max-tilt", type=float, default=MAX_TILT_DEG, help="각도를 모를 때 최대 기울기(도)")
    ap.add_argument("--dem", nargs="*", help="DEM IMG 파일들 (해발 고도 변환과 지형 반영에 씀)")
    ap.add_argument("--geoid", type=float, help="타원체 고도일 때 지오이드 높이(m)")
    a = ap.parse_args()

    dem = None
    if a.dem:
        from dem import Dem
        dem = Dem(a.dem)

    with open(a.detection_json, encoding="utf-8") as f:
        frame = json.load(f)
    if not frame.get("detections"):
        print("탐지 없음")
        return
    for det in frame["detections"]:
        try:
            pos, info = to_position(frame, det, a.alt, a.max_tilt, dem, a.geoid)
        except ValueError as e:
            print(f"{det.get('id')}: 변환 실패 - {e}")
            continue
        print(f"{det.get('id')}: {json.dumps(pos, ensure_ascii=False)}")
        print(f"   밑동 픽셀 {info['pixel']}, 드론에서 {info['ground_offset_m']}m, "
              f"지면 위 고도 {info['alt_agl_m']}m, 내려다보는 각 {info['depression_deg']}도")
        for note in info["assumptions"]:
            print(f"   가정: {note}")


if __name__ == "__main__":
    sys.exit(main())
