# -*- coding: utf-8 -*-
"""관제 대시보드 서버 (WBS 5.5).

파이프라인(src/pipeline/run.py)이 내는 알람 폴더를 읽어 지도에 보여 준다.
외부 패키지 없이 돈다. 지도 라이브러리(Leaflet 1.9.4)는 vendor/ 에 들어 있어
인터넷이 없어도 화면은 뜬다(배경 지도 타일만 빠진다).

    python src/dashboard/server.py --out results/pipeline --geo-dir data/geo_moraegogae
    → 브라우저에서 http://localhost:8800

파이프라인을 돌리기 전이나 도는 중에 켜 두어도 된다. 화면이 1초마다 알람 폴더를 다시 읽는다.

배경 지도
    - 기본: OpenStreetMap
    - 환경변수 VWORLD_KEY 가 있으면 V-World 위성 지도와 일반 지도를 고를 수 있다
      (인증키에 http://localhost 가 등록되어 있어야 한다)

화면에 보이는 것
    - 알람 위치 마커와 오차 반경 원 (판정에 따라 색: 확정 빨강, 모호 주황, 미확정 회색)
    - 드론 위치와 알람까지 잇는 점선 (탐지 프레임의 camera 값)
    - 해설 카드: 판정, 설명(생성 중이면 표시), 반경 안 용도 비율, 건물 수, 주변 경사,
      알람·설명까지 걸린 시간, 프레임 사진
    - 용도지역 폴리곤 (--geo-dir 를 주면. 켜고 끌 수 있음)
"""
import argparse
import json
import math
import mimetypes
import os
import sys
import urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = (37.736790, 128.906239)    # 모래고개입구


# ---------- EPSG:5179 → 위도·경도 (src/geo/query.py to_5179 의 역변환) ----------
def from_5179(x, y):
    a = 6378137.0
    f = 1 / 298.257222101
    e2 = f * (2 - f)
    k0, lon0, lat0, fe, fn = 0.9996, math.radians(127.5), math.radians(38), 1e6, 2e6

    def m(p):
        return a * ((1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * p
                    - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * math.sin(2 * p)
                    + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * math.sin(4 * p)
                    - (35 * e2**3 / 3072) * math.sin(6 * p))

    ep2 = e2 / (1 - e2)
    mm = m(lat0) + (y - fn) / k0
    mu = mm / (a * (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256))
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    p1 = (mu + (3 * e1 / 2 - 27 * e1**3 / 32) * math.sin(2 * mu)
          + (21 * e1**2 / 16 - 55 * e1**4 / 32) * math.sin(4 * mu)
          + (151 * e1**3 / 96) * math.sin(6 * mu) + (1097 * e1**4 / 512) * math.sin(8 * mu))
    c1 = ep2 * math.cos(p1) ** 2
    t1 = math.tan(p1) ** 2
    n1 = a / math.sqrt(1 - e2 * math.sin(p1) ** 2)
    r1 = a * (1 - e2) / (1 - e2 * math.sin(p1) ** 2) ** 1.5
    d = (x - fe) / (n1 * k0)
    lat = p1 - (n1 * math.tan(p1) / r1) * (d * d / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1 * c1 - 9 * ep2) * d**4 / 24
                                           + (61 + 90 * t1 + 298 * c1 + 45 * t1 * t1 - 252 * ep2 - 3 * c1 * c1) * d**6 / 720)
    lon = lon0 + (d - (1 + 2 * t1 + c1) * d**3 / 6
                  + (5 - 2 * c1 + 28 * t1 - 3 * c1 * c1 + 8 * ep2 + 24 * t1 * t1) * d**5 / 120) / math.cos(p1)
    return math.degrees(lon), math.degrees(lat)


def landuse_4326(path):
    """용도지역 폴리곤을 지도에 그릴 수 있게 위도·경도로 바꾼다 (한 번만)."""
    gj = json.load(open(path, encoding="utf-8"))

    def conv(coords):
        if isinstance(coords[0], (int, float)):
            lon, lat = from_5179(coords[0], coords[1])
            return [round(lon, 6), round(lat, 6)]
        return [conv(c) for c in coords]

    feats = []
    for f in gj["features"]:
        g = f["geometry"]
        feats.append({"type": "Feature",
                      "properties": {"name": (f.get("properties") or {}).get("uname")},
                      "geometry": {"type": g["type"], "coordinates": conv(g["coordinates"])}})
    return {"type": "FeatureCollection", "features": feats}


# ---------- 알람 읽기 ----------
def read_state(out, frames_dir):
    """알람 목록과, 알람마다 드론 위치·사진 경로를 붙여 돌려준다."""
    out = Path(out)
    cams = {}       # detection_id → (camera, frame 이름, 사진 경로)
    for f in (out / "detections").glob("*.json"):
        try:
            fr = json.load(open(f, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        img = fr.get("image")
        for d in fr.get("detections") or []:
            cams[d.get("id")] = (fr.get("camera"), f.stem, img, d, fr.get("image_size"))
    alerts, extra = [], {}
    for f in sorted((out / "alerts").glob("*.json")):
        try:
            al = json.load(open(f, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue    # 쓰는 중인 파일은 다음 번에 읽는다
        alerts.append(al)
        cam, stem, img, det, size = cams.get(al.get("detection_id"), (None,) * 5)
        box = None
        if det and size and det.get("bbox"):
            x, y, w, h = det["bbox"]
            W, H = size["width"], size["height"]
            box = {"left": x / W, "top": y / H, "width": w / W, "height": h / H,
                   "score": det.get("score"), "aspect": W / H}
        extra[al["alert_id"]] = {
            "drone": {"lat": cam["lat"], "lon": cam["lon"], "heading": cam.get("heading_deg"),
                      "alt": cam.get("alt_m"), "alt_ref": cam.get("alt_ref")}
            if cam and cam.get("lat") is not None else None,
            "image": f"/api/image/{urllib.parse.quote(stem)}" if stem and _image_path(img, frames_dir) else None,
            "box": box,
        }
    return {"alerts": alerts, "extra": extra}


def _image_path(img, frames_dir):
    if not img:
        return None
    p = Path(img)
    if p.is_file():
        return p
    if frames_dir and (Path(frames_dir) / p.name).is_file():
        return Path(frames_dir) / p.name
    return None


class Handler(BaseHTTPRequestHandler):
    cfg = {}

    def log_message(self, *args):     # 1초마다 오는 요청으로 화면이 차지 않게 한다
        pass

    def _send(self, code, body, ctype, cache=False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(200, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _file(self, path, cache=False):
        path = Path(path)
        if not path.is_file():
            return self._send(404, b"not found", "text/plain")
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self._send(200, path.read_bytes(), ctype, cache)

    def do_GET(self):
        c = self.cfg
        u = urllib.parse.urlparse(self.path)
        p = urllib.parse.unquote(u.path)
        if p in ("/", "/index.html"):
            return self._file(HERE / "index.html")
        if p.startswith("/vendor/"):
            target = (HERE / p.lstrip("/")).resolve()
            if HERE / "vendor" not in target.parents:
                return self._send(403, b"forbidden", "text/plain")
            return self._file(target, cache=True)
        if p == "/api/config":
            return self._json({"center": c["center"], "vworld_key": c["vworld_key"],
                               "has_landuse": c["landuse"] is not None})
        if p == "/api/alerts":
            return self._json(read_state(c["out"], c["frames"]))
        if p == "/api/landuse":
            return self._json(c["landuse"] or {"type": "FeatureCollection", "features": []})
        if p.startswith("/api/image/"):
            stem = p[len("/api/image/"):]
            f = Path(c["out"]) / "detections" / f"{Path(stem).name}.json"
            if not f.is_file():
                return self._send(404, b"not found", "text/plain")
            img = _image_path(json.load(open(f, encoding="utf-8")).get("image"), c["frames"])
            return self._file(img, cache=True) if img else self._send(404, b"not found", "text/plain")
        self._send(404, b"not found", "text/plain")


def main():
    ap = argparse.ArgumentParser(description="관제 대시보드 (WBS 5.5)")
    ap.add_argument("--out", default="results/pipeline", help="파이프라인 출력 폴더")
    ap.add_argument("--frames", help="사진 폴더 (탐지 JSON에 파일 이름만 있을 때)")
    ap.add_argument("--geo-dir", help="landuse_5179.geojson 이 있는 폴더 (용도지역을 지도에 그림)")
    ap.add_argument("--port", type=int, default=8800)
    a = ap.parse_args()

    lu = None
    if a.geo_dir and (Path(a.geo_dir) / "landuse_5179.geojson").is_file():
        lu = landuse_4326(Path(a.geo_dir) / "landuse_5179.geojson")
        print(f"[준비] 용도지역 폴리곤 {len(lu['features'])}개")
    Handler.cfg = {"out": a.out, "frames": a.frames, "landuse": lu,
                   "center": list(SITE), "vworld_key": os.environ.get("VWORLD_KEY", "").strip() or None}
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"[대시보드] http://localhost:{a.port}  (알람 폴더: {Path(a.out) / 'alerts'})")
    print("           끄려면 Ctrl+C")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
