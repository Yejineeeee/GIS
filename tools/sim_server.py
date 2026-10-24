# -*- coding: utf-8 -*-
"""시뮬레이터 페이지를 띄우고 프레임을 받아 저장하는 작은 서버.

브라우저에서 캔버스를 PNG 로 뽑아도 디스크로 보낼 방법이 없다. 내려받기는
사용자 승인이 걸리거나 어디에 떨어졌는지 알기 어렵다. 그래서 POST 를 받아
바로 저장한다.

    python sim_server.py 폴더 [포트]

    GET  /            시뮬레이터 페이지
    POST /frame       {"png": "data:image/png;base64,...", "meta": {...}}
                      -> frames/sim_<번호>.png 와 .json 으로 저장
"""
import base64
import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler

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
        if self.path != "/frame":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception as e:
            self.send_error(400, str(e))
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

        out = json.dumps({"saved": stem}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, fmt, *args):
        if "POST" in (args[0] if args else ""):
            super().log_message(fmt, *args)


if __name__ == "__main__":
    print(f"{ROOT} 를 http://localhost:{PORT} 로 띄운다")
    print(f"프레임은 {FRAMES} 에 저장한다")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
