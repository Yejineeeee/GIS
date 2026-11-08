# -*- coding: utf-8 -*-
"""연기 밑동이 능선에 가려졌을 수 있는지 지형으로 판단한다.

사진 한 장으로 좌표를 낼 때는 마스크 맨 아래 픽셀을 밑동으로 보고, 그 픽셀의 시선이
지면과 처음 만나는 점을 발화점으로 쓴다(transform.to_position). 그런데 밑동이 앞 능선
뒤 골짜기에 숨어 있으면, 보이는 맨 아래 픽셀은 능선 위로 올라온 연기다. 그 시선은
능선을 살짝 넘어 훨씬 뒤의 땅에 닿고, 좌표가 수백 m 멀리 찍힌다.

여기서는 그 시선을 따라가며 발화점 후보를 다시 찾는다. 후보는 두 가지다.
    - 시선이 땅에 닿은 점 (지금 방식의 답)
    - 시선 아래, 카메라에서 안 보이는 땅 가운데, 거기서 연기가 MAX_COLUMN_M 이하로만
      올라와도 이 시선에 걸리는 곳 (능선 뒤에 숨은 밑동)
뒤쪽 후보가 MIN_SPAN_M 이상 이어지면 가림 가능으로 본다. 후보 전체를 덮도록 위치를
가운데로 옮기고 반경을 키운다.

    from occlusion import hidden_span, adjust
    span = hidden_span(cam_lat, cam_lon, alt_agl, heading, pitch, hfov, px, py, W, H, dem)
    # None 이면 가림 없음. 아니면 {"near_m", "hit_m", "length_m", "lat", "lon", "half_m"}
    #   near_m: 숨은 후보 중 가장 가까운 곳, hit_m: 시선이 땅에 닿은 곳 (드론에서 수평 거리)
    #   lat, lon: 둘의 가운데, half_m: 반경에 더할 값(후보 구간 길이의 절반)

    position, span = adjust(frame, info, position, dem)   # to_position 결과에 바로 반영

transform.py 는 고치지 않는다. adjust 가 position 을 (lat, lon) 으로 옮기고
error_radius_m 에 half_m 을 더한다. 판정은 그 position 으로 한다.
"""
import math

from query import to_5179

STEP_M = 5.0          # 시선을 따라가는 간격
MAX_RANGE_M = 4000.0  # 시선을 어디까지 따라갈지
MAX_COLUMN_M = 40.0   # 이 높이까지 올라온 연기는 이 시선에 보일 수 있다고 본다 (초기 발화 기둥 높이 가정)
MIN_SPAN_M = 20.0     # 숨은 후보가 이보다 짧게 이어지면 무시한다
R = 6378137.0


def _move(lat, lon, e, n):
    return (lat + math.degrees(n / R), lon + math.degrees(e / (R * math.cos(math.radians(lat)))))


def _ray_dir(px, py, W, H, heading, pitch, hfov):
    """수평 1m 당 (동, 북, 위)."""
    f = (W / 2) / math.tan(math.radians(hfov) / 2)
    dx, dy = px - W / 2, py - H / 2
    p, h = math.radians(pitch), math.radians(heading)
    fwd = f * math.cos(p) + dy * math.sin(p)
    up = f * math.sin(p) - dy * math.cos(p)
    e = fwd * math.sin(h) + dx * math.cos(h)
    n = fwd * math.cos(h) - dx * math.sin(h)
    horiz = math.hypot(e, n)
    return e / horiz, n / horiz, up / horiz


def hidden_span(lat, lon, alt_agl, heading, pitch, hfov, px, py, W, H, dem):
    """능선 뒤에 숨은 밑동 후보 구간. 가림이 없거나 지형을 모르면 None."""
    g0 = dem.height(*to_5179(lon, lat))
    if g0 is None:
        return None
    ue, un, uu = _ray_dir(px, py, W, H, heading, pitch, hfov)
    if uu >= 0:
        return None
    z0 = g0 + alt_agl
    max_slope = -1e9          # 지금까지 본 땅 가운데 가장 높이 보이는 기울기 (가시성)
    hidden = []               # 숨은 후보 거리들
    hit = None
    d = 0.0
    while d < MAX_RANGE_M:
        d += STEP_M
        la, lo = _move(lat, lon, ue * d, un * d)
        g = dem.height(*to_5179(lo, la))
        if g is None:
            break
        slope = (g - z0) / d
        visible = slope >= max_slope
        max_slope = max(max_slope, slope)
        gap = (z0 + uu * d) - g                     # 시선이 땅보다 얼마나 위에 있나
        if gap <= 0:
            hit = d
            break
        if not visible and gap <= MAX_COLUMN_M:
            hidden.append(d)
    if hit is None or not hidden:
        return None
    # 숨은 후보 가운데 가장 긴 연속 구간
    runs, cur = [], [hidden[0]]
    for x in hidden[1:]:
        if x - cur[-1] <= 2 * STEP_M:
            cur.append(x)
        else:
            runs.append(cur)
            cur = [x]
    runs.append(cur)
    best = max(runs, key=lambda r: r[-1] - r[0])
    if best[-1] - best[0] < MIN_SPAN_M:
        return None
    near = best[0]
    mid = (near + hit) / 2
    la, lo = _move(lat, lon, ue * mid, un * mid)
    return {"near_m": round(near, 1), "hit_m": round(hit, 1), "length_m": round(hit - near, 1),
            "lat": la, "lon": lo, "half_m": round((hit - near) / 2, 1)}


def adjust(frame, info, position, dem):
    """to_position 결과에 가림을 반영한다. (position, span) 을 돌려준다.

    가림이 없거나 지형, 카메라 각도를 모르면 position 을 그대로, span 은 None.
    가림이 있으면 위치를 후보 구간 가운데로 옮기고 반경에 half_m 을 더한 새 position.
    """
    cam = frame.get("camera") or {}
    size = frame.get("image_size") or {}
    need = ("lat", "lon", "heading_deg", "pitch_deg", "hfov_deg")
    if dem is None or any(cam.get(k) is None for k in need) or not size:
        return position, None
    px, py = info["pixel"]
    span = hidden_span(cam["lat"], cam["lon"], info["alt_agl_m"], cam["heading_deg"], cam["pitch_deg"],
                       cam["hfov_deg"], px, py, size["width"], size["height"], dem)
    if not span:
        return position, None
    pos = dict(position)
    pos["lat"], pos["lon"] = round(span["lat"], 7), round(span["lon"], 7)
    pos["error_radius_m"] = round(position["error_radius_m"] + span["half_m"], 1)
    return pos, span
