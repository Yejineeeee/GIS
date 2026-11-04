# -*- coding: utf-8 -*-
"""시뮬레이터 페이지를 띄우고 프레임을 받아 저장하는 작은 서버.

브라우저에서 캔버스를 PNG 로 뽑아도 디스크로 보낼 방법이 없다. 내려받기는
사용자 승인이 걸리거나 어디에 떨어졌는지 알기 어렵다. 그래서 POST 를 받아
바로 저장한다.

    python sim_server.py 폴더 [포트]

    GET  /            시뮬레이터 페이지
    POST /frame       {"png": "data:image/png;base64,...", "meta": {...}}
                      -> frames/sim_<번호>.png 와 .json 으로 저장
    POST /terrain     지형 높이 격자 -> terrain.json 으로 저장 (src/geo/sim_terrain.py)
"""
import base64
import json
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8712
FRAMES = os.path.join(ROOT, "frames")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def end_headers(self):
        # 같은 출처에서만 쓰지만, 다른 포트에서 열어도 되게 열어 둔다
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_POST(self):
        if self.path not in ("/frame", "/terrain"):
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception as e:
            self.send_error(400, str(e))
            return

        if self.path == "/terrain":
            # 지형 높이 격자 (index.html 의 terrainGrid). 한 벌만 두고 덮어쓴다
            with open(os.path.join(ROOT, "terrain.json"), "w", encoding="utf-8") as f:
                json.dump(body, f)
            self._json({"saved": "terrain.json"})
            return

        os.makedirs(FRAMES, exist_ok=True)
        i = len([f for f in os.listdir(FRAMES) if f.endswith(".png")])
        stem = f"sim_{i:04d}"
        png = body.get("png", "")
        if png.startswith("data:"):
            png = png.split(",", 1)[1]
        with open(os.path.join(FRAMES, stem + ".png"), "wb") as f:
            f.write(base64.b64decode(png))
        meta = body.get("meta", {})
        meta["image"] = stem + ".png"
        with open(os.path.join(FRAMES, stem + ".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        self._json({"saved": stem})

    def _json(self, obj):
        out = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, fmt, *args):
        """POST 만 찍는다. 타일 GET 이 많아 그대로 두면 가려진다.

        args[0] 이 요청 줄일 때도 있고 상태 코드일 때도 있다. 404 처럼 코드가
        들어오면 문자열로 보는 순간 TypeError 가 났고, 그 예외가 응답을 쓰다
        말고 끊어 브라우저의 fetch 가 영영 안 끝났다. 한 번 그렇게 되면
        한 줄짜리 서버라 다음 요청도 못 받는다. 문자열로 바꿔서 본다.
        """
        first = str(args[0]) if args else ""
        if "POST" in first:
            super().log_message(fmt, *args)


if __name__ == "__main__":
    print(f"{ROOT} 를 http://localhost:{PORT} 로 띄운다")
    print(f"프레임은 {FRAMES} 에 저장한다")
    # 한 줄짜리 서버면 한 연결이 막힐 때 전체가 선다. 프레임을 받는 중에
    # 브라우저가 다른 요청을 보내므로 연결마다 따로 받는다.
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
