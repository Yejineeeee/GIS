# -*- coding: utf-8 -*-
"""알람 지연과 설명 지연 측정 (WBS 7.4).

7.3(tools/measure_device.py)은 사진을 쉬지 않고 몰아 넣어 기기가 버티는지 봤다. 여기서는 드론처럼
정해진 간격(기본 1.5초)으로 사진을 한 장씩 넣으면서, 관제 요원이 화면에서 기다리는 시간을 잰다.

    사진 생김 ──(알람 지연)──▶ 대시보드에 알람 ──(설명 지연)──▶ 대시보드에 설명

어떻게 재나
    1) run.py 를 지켜보기 모드(--watch)로 띄운다. 사진 폴더는 --watch-dir (기본 sim/frames_latency)
    2) 대시보드 서버(src/dashboard/server.py)를 같은 결과 폴더로 띄운다
    3) --src 의 사진과 JSON 을 --interval 초마다 한 장씩 --watch-dir 에 넣는다 (시뮬레이터 대신)
    4) 그동안 두 곳을 지켜본다
       - alerts.jsonl: 파이프라인이 알람을 낼 때마다 한 줄. 0.05초마다 읽어 "파일에 나온 시각"을 적는다
       - 대시보드 /api/alerts: 브라우저처럼 --poll 초(기본 1초)마다 읽어 "화면에 뜬 시각"을 적는다
    5) 탐지 번호(detection_id)로 그 사진을 넣은 시각과 이어 지연을 낸다

    # 알람 지연만 (VLM 없이 설명은 바로 붙음)
    python tools/measure_latency.py --report results/latency_fake.md -- \\
        --weights best.pt --geo-dir data/geo_jagok --sim-terrain sim/terrain_jagok3d.json --vlm fake

    # 설명 지연까지 (llama-server 를 먼저 띄운다. 잴 때마다 새로 띄워야 이미지 캐시가 없다)
    python tools/measure_latency.py --report results/latency_gguf.md -- \\
        --weights best.pt --geo-dir data/geo_jagok --sim-terrain sim/terrain_jagok3d.json --vlm gguf

-- 뒤는 run.py 인자다. --watch, --frames, --out, --idle-sec 는 이 스크립트가 붙인다.
결과: --report 의 마크다운 표와 같은 이름의 .json
"""
import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def frame_key(detection_id):
    """'SCENARIO_VW3D-24-0' → ('SCENARIO_VW3D', 24). 탐지 번호는 영상-프레임-순번이다."""
    try:
        vid, fid, _ = str(detection_id).rsplit("-", 2)
        return vid, int(fid)
    except ValueError:
        return None


def stats(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    p90 = xs[min(len(xs) - 1, int(round(0.9 * (len(xs) - 1))))]
    return {"n": len(xs), "median": round(statistics.median(xs), 2), "p90": round(p90, 2),
            "max": round(xs[-1], 2), "min": round(xs[0], 2)}


def fmt(s):
    return "{:.2f}초".format(s) if s is not None else "-"


class Tail(threading.Thread):
    """alerts.jsonl 을 0.05초마다 읽어 새 줄마다 (읽은 시각, 알람)을 쌓는다."""

    def __init__(self, path):
        super().__init__(daemon=True)
        self.path, self.rows, self.stop, self.pos, self.buf = Path(path), [], threading.Event(), 0, ""

    def read_once(self):
        if not self.path.is_file():
            return
        with open(self.path, encoding="utf-8") as f:
            f.seek(self.pos)
            chunk = f.read()
            self.pos = f.tell()
        now = time.time()
        self.buf += chunk
        *lines, self.buf = self.buf.split("\n")
        for line in lines:
            if line.strip():
                try:
                    self.rows.append((now, json.loads(line)))
                except ValueError:
                    pass

    def run(self):
        while not self.stop.is_set():
            self.read_once()
            time.sleep(0.05)
        self.read_once()


class Dash(threading.Thread):
    """대시보드 /api/alerts 를 브라우저처럼 poll 초마다 읽어, 탐지 번호별로 알람과 설명이 처음 보인 시각을 적는다."""

    def __init__(self, url, poll):
        super().__init__(daemon=True)
        self.url, self.poll, self.stop = url, poll, threading.Event()
        self.alert_seen, self.desc_seen, self.errors = {}, {}, 0

    def read_once(self):
        try:
            with urllib.request.urlopen(self.url + "/api/alerts", timeout=5) as r:
                st = json.load(r)
        except Exception:
            self.errors += 1
            return
        now = time.time()
        for al in st.get("alerts", []):
            d = al.get("detection_id")
            self.alert_seen.setdefault(d, now)
            if al.get("description"):
                self.desc_seen.setdefault(d, now)

    def run(self):
        while not self.stop.is_set():
            t0 = time.time()
            self.read_once()
            self.stop.wait(max(0.0, self.poll - (time.time() - t0)))


def wait_http(url, sec=30):
    end = time.time() + sec
    while time.time() < end:
        try:
            urllib.request.urlopen(url + "/api/config", timeout=2).read()
            return True
        except Exception:
            time.sleep(0.3)
    return False


def main():
    argv = sys.argv[1:]
    run_args = []
    if "--" in argv:
        i = argv.index("--")
        argv, run_args = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description="알람 지연과 설명 지연 측정 (WBS 7.4)")
    ap.add_argument("--src", default="sim/frames_scenario", help="넣을 사진 폴더 (sim_NNNN.png + .json)")
    ap.add_argument("--interval", type=float, default=1.5, help="사진 간격 초 (기본 1.5, 드론 프레임 간격)")
    ap.add_argument("--jitter", type=float, default=0.3,
                    help="간격을 ±이만큼 무작위로 흔든다 (실제 드론도 일정하지 않고, 화면 1초 주기와 박자가 맞물리지 않게)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--watch-dir", default="sim/frames_latency", help="run.py 가 지켜볼 폴더 (시작할 때 비운다)")
    ap.add_argument("--out", default="results/latency_run", help="run.py 결과 폴더")
    ap.add_argument("--dash-port", type=int, default=8830, help="측정용 대시보드 포트 (0 이면 대시보드 없이 파일만 잼)")
    ap.add_argument("--poll", type=float, default=1.0, help="대시보드를 읽는 간격 초 (화면과 같은 1초)")
    ap.add_argument("--idle-sec", type=float, default=10, help="마지막 사진 뒤 이만큼 새 사진이 없으면 run.py 가 끝냄")
    ap.add_argument("--cores", type=int, default=4, help="run.py 를 묶을 코어 수 (리눅스 taskset, 0 이면 묶지 않음)")
    ap.add_argument("--report", default="results/latency.md")
    a = ap.parse_args(argv)
    if not run_args:
        ap.error("-- 뒤에 run.py 인자를 준다 (--weights, --geo-dir, --vlm ...)")
    for bad in ("--watch", "--frames", "--out", "--idle-sec", "--interval"):
        if bad in run_args:
            ap.error(f"{bad} 는 이 스크립트가 붙인다. -- 뒤에서 빼 달라")

    src, wdir, out = (ROOT / a.src).resolve(), (ROOT / a.watch_dir).resolve(), (ROOT / a.out).resolve()
    pngs = sorted(src.glob("*.png"))
    if not pngs:
        sys.exit(f"{src} 에 사진이 없다")
    wdir.mkdir(parents=True, exist_ok=True)
    for p in list(wdir.glob("*.png")) + list(wdir.glob("*.json")) + list(wdir.glob("STOP")):
        p.unlink()
    if (out / "alerts.jsonl").exists():
        (out / "alerts.jsonl").unlink()

    # 1) run.py 지켜보기 모드
    cmd = [sys.executable, str(ROOT / "src/pipeline/run.py"), *run_args,
           "--watch", "--frames", str(wdir), "--out", str(out), "--idle-sec", str(a.idle_sec)]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if a.cores and platform.system() == "Linux" and shutil.which("taskset"):
        cmd = ["taskset", "-c", f"0-{a.cores - 1}", *cmd]
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[k] = str(a.cores)
    print("[측정] " + " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    log, ready = [], threading.Event()

    def pump():
        for line in proc.stdout:
            log.append(line.rstrip("\n"))
            print("  | " + line.rstrip("\n"))
            if "새 사진이 오기를 기다린다" in line:
                ready.set()
    threading.Thread(target=pump, daemon=True).start()
    if not ready.wait(180):
        proc.kill()
        sys.exit("run.py 가 지켜보기를 시작하지 못했다 (위 출력 확인)")

    # 2) 대시보드
    dash_proc, dash = None, None
    if a.dash_port:
        url = f"http://127.0.0.1:{a.dash_port}"
        dash_proc = subprocess.Popen([sys.executable, str(ROOT / "src/dashboard/server.py"), "--out", str(out),
                                      "--frames", str(wdir), "--port", str(a.dash_port)],
                                     cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not wait_http(url):
            dash_proc.kill()
            proc.kill()
            sys.exit(f"대시보드가 {url} 에서 뜨지 않았다 (포트가 쓰이고 있는지 확인)")
        dash = Dash(url, a.poll)
        dash.start()
    tail = Tail(out / "alerts.jsonl")
    tail.start()

    # 3) 사진을 간격대로 넣는다. 사진을 먼저, JSON 을 나중에 (시뮬레이터와 같은 순서)
    import random
    rnd = random.Random(a.seed)
    fed = {}
    t_start = time.time()
    target = t_start
    for i, png in enumerate(pngs):
        if i:
            target += max(0.1, a.interval + rnd.uniform(-a.jitter, a.jitter))
        time.sleep(max(0.0, target - time.time()))
        side = png.with_suffix(".json")
        meta = json.load(open(side, encoding="utf-8")) if side.is_file() else {}
        shutil.copyfile(png, wdir / png.name)            # copyfile: 수정 시각을 지금으로 (run.py 가 이 시각을 씀)
        t = time.time()
        tmp = wdir / (png.stem + ".json.tmp")
        shutil.copyfile(side, tmp)
        tmp.replace(wdir / side.name)
        fed[(meta.get("video_id"), meta.get("frame_id", i))] = t
    late = max(0.0, time.time() - target)
    print(f"[측정] {len(pngs)}장을 {a.interval:g}초 간격으로 넣음 (넣기 밀림 {late:.2f}초). run.py 가 끝나기를 기다린다")

    try:
        proc.wait(timeout=900)
    except subprocess.TimeoutExpired:
        (wdir / "STOP").touch()
        proc.wait(timeout=300)
    time.sleep(max(1.5, a.poll * 1.5))                    # 마지막 설명이 화면에 한 번 더 읽히게
    tail.stop.set()
    tail.join()
    if dash:
        dash.stop.set()
        dash.join()
        dash_proc.terminate()

    # 4) 정리
    rows = tail.rows
    first_line, desc_line, info = {}, {}, {}
    for t, al in rows:
        d = al.get("detection_id")
        first_line.setdefault(d, t)
        info.setdefault(d, {"alert_id": al.get("alert_id"), "verdict": al.get("location_verdict"),
                            "radius_m": (al.get("position") or {}).get("error_radius_m")})
        if al.get("description"):
            desc_line.setdefault(d, t)
    versions = []
    for d, t_al in sorted(first_line.items(), key=lambda kv: kv[1]):
        k = frame_key(d)
        t0 = fed.get(k)
        dv = {"detection_id": d, **info[d],
              "photo_to_alert_file": round(t_al - t0, 3) if t0 else None,
              "photo_to_alert_screen": round(dash.alert_seen[d] - t0, 3) if dash and t0 and d in dash.alert_seen else None,
              "alert_to_desc_file": round(desc_line[d] - t_al, 3) if d in desc_line else None,
              "alert_to_desc_screen": round(dash.desc_seen[d] - dash.alert_seen[d], 3)
              if dash and d in dash.desc_seen and d in dash.alert_seen else None,
              "photo_to_desc_screen": round(dash.desc_seen[d] - t0, 3) if dash and t0 and d in dash.desc_seen else None}
        versions.append(dv)

    by_alert = {}
    for v in versions:
        by_alert.setdefault(v["alert_id"], []).append(v)
    firsts = []          # 알람마다 처음 낸 판(관제 요원이 처음 보는 알람)과 처음 붙은 설명
    for aid, vs in by_alert.items():
        f = vs[0]
        d_first = next((v for v in vs if v["photo_to_desc_screen"] is not None), None)
        t0 = fed.get(frame_key(f["detection_id"]))
        first_desc_screen = (dash.desc_seen[d_first["detection_id"]] - t0) if dash and d_first and t0 else None
        firsts.append({"alert_id": aid, "updates": len(vs) - 1,
                       "photo_to_alert_screen": f["photo_to_alert_screen"],
                       "photo_to_alert_file": f["photo_to_alert_file"],
                       "photo_to_first_desc_screen": round(first_desc_screen, 3) if first_desc_screen else None,
                       "last_has_desc": vs[-1]["alert_to_desc_file"] is not None})

    S = {
        "사진 → 알람 (파일)": stats(v["photo_to_alert_file"] for v in versions),
        "사진 → 알람 (화면)": stats(v["photo_to_alert_screen"] for v in versions),
        "알람 → 설명 (파일)": stats(v["alert_to_desc_file"] for v in versions),
        "알람 → 설명 (화면)": stats(v["alert_to_desc_screen"] for v in versions),
        "사진 → 설명 (화면)": stats(v["photo_to_desc_screen"] for v in versions),
    }
    vlm = run_args[run_args.index("--vlm") + 1] if "--vlm" in run_args else "none"
    no_desc = sum(1 for v in versions if v["alert_to_desc_file"] is None)
    watch_lines = [l.strip() for l in log if "사진 생김" in l or "프레임 " in l and "알람 " in l]

    md = [f"# 알람 지연과 설명 지연 측정 결과 (7.4)", "",
          f"측정: {time.strftime('%Y-%m-%d %H:%M')} / `tools/measure_latency.py`", "",
          "## 조건", "", "| 항목 | 값 |", "|---|---|",
          f"| 기기 | {platform.platform()} / CPU {os.cpu_count()}개 보임, run.py {a.cores or '제한 없음'}코어 |",
          f"| 사진 | `{a.src}` {len(pngs)}장, {a.interval:g}초 ±{a.jitter:g}초 간격 (넣기 밀림 {late:.2f}초) |",
          f"| VLM | `{vlm}` |",
          f"| 화면 | 대시보드 `/api/alerts` 를 {a.poll:g}초마다 읽음 (브라우저와 같은 주기) |" if dash else "| 화면 | 재지 않음 |",
          f"| run.py 인자 | {' '.join(run_args)} |", "",
          "## 지연 (알람 판마다)", "",
          "알람 판 = 새 알람이나 갱신(같은 alert_id, 다른 탐지)마다 하나. 설명이 붙기 전에 다음 갱신이 오면 그 판의 설명은 버린다.", "",
          "| 구간 | 개수 | 중앙값 | 90% | 최대 |", "|---|---|---|---|---|"]
    for k, s in S.items():
        md.append(f"| {k} | {s['n'] if s else 0} | {fmt(s and s['median'])} | {fmt(s and s['p90'])} | {fmt(s and s['max'])} |")
    md += ["", f"알람 판 {len(versions)}개 중 설명이 붙지 않고 다음 갱신으로 넘어간 판: {no_desc}개", "",
           "## 알람별 (관제 요원이 처음 보는 시점)", "",
           "| 알람 | 갱신 | 사진 → 첫 알람 (화면) | 사진 → 첫 설명 (화면) | 마지막 판 설명 |", "|---|---|---|---|---|"]
    for f in firsts:
        md.append(f"| {f['alert_id']} | {f['updates']}번 | {fmt(f['photo_to_alert_screen'])} | "
                  f"{fmt(f['photo_to_first_desc_screen'])} | {'붙음' if f['last_has_desc'] else '없음'} |")
    md += ["", "## 판별 기록", "",
           "| 탐지 | 알람 | 판정 | 반경 | 사진→알람(파일) | 사진→알람(화면) | 알람→설명(파일) | 사진→설명(화면) |",
           "|---|---|---|---|---|---|---|---|"]
    for v in versions:
        md.append(f"| {v['detection_id']} | {v['alert_id']} | {v['verdict']} | {v['radius_m']}m | "
                  f"{fmt(v['photo_to_alert_file'])} | {fmt(v['photo_to_alert_screen'])} | "
                  f"{fmt(v['alert_to_desc_file'])} | {fmt(v['photo_to_desc_screen'])} |")
    if watch_lines:
        md += ["", "## run.py 요약", "", "```", *watch_lines[-4:], "```"]
    if dash and dash.errors:
        md += ["", f"대시보드 읽기 실패 {dash.errors}번"]
    rep = ROOT / a.report
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text("\n".join(md) + "\n", encoding="utf-8")
    rep.with_suffix(".json").write_text(json.dumps(
        {"interval": a.interval, "frames": len(pngs), "vlm": vlm, "summary": S, "alerts": firsts,
         "jitter": a.jitter, "versions": versions, "feed_late_s": round(late, 3)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(md))
    print(f"\n[측정] 결과: {rep} , {rep.with_suffix('.json')}")
    return 0 if proc.returncode == 0 else proc.returncode


if __name__ == "__main__":
    sys.exit(main())
