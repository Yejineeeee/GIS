# -*- coding: utf-8 -*-
"""현장 기기 동작 조건 측정 (WBS 7.3). 메모리와 처리 시간.

기준 환경(docs/device_spec.md): CPU 4코어, 메모리 8GB(스왑 없음), GPU 없음.
파이프라인(src/pipeline/run.py)을 기준 코어 수에 묶어 끝까지 돌리고, 도는 동안 메모리를 재고,
끝나면 알람과 탐지 기록에서 단계별 시간을 모은다.

    # 1) 부품별 메모리 (탐지 모델, 지리 자료, 지형 격자를 하나씩 올리며 잰다)
    # 2) 파이프라인 전체 (run.py 를 그대로 돌린다. -- 뒤는 run.py 인자)
    python tools/measure_device.py --cores 4 --mem-gb 8 --report results/device_measure.md -- \\
        --weights best.pt --frames sim/frames_scenario --geo-dir data/geo_jagok \\
        --sim-terrain sim/terrain_jagok3d.json --vlm gguf --out results/device_run

VLM 을 gguf 로 재려면 llama-server 를 같은 기기에서 4스레드로 먼저 띄운다. 이 스크립트는 llama-server
프로세스 메모리도 찾아 같이 잰다(파이프라인 + llama-server 합이 8GB 안이어야 한다).

    llama-server -m Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf --mmproj mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf \\
        -t 4 -c 4096 --host 127.0.0.1 --port 8099

코어 제한
    리눅스(WSL 포함)는 taskset 으로 0..cores-1 코어에 묶고, OMP/MKL 스레드 수도 cores 로 맞춘다.
    윈도우는 psutil 의 cpu_affinity 로 묶는다. llama-server 는 -t 4 로 띄운다.
메모리 제한
    걸지 않고 최대 사용량을 재서 기준과 견준다. WSL 에서 기기 전체를 기준에 맞추려면
    %UserProfile%\\.wslconfig 에 [wsl2] memory=8GB processors=4 swap=0 을 넣고 wsl --shutdown 뒤 다시 켠다.

결과: --report 의 마크다운 표와 같은 이름의 .json
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GB = 2 ** 30
MB = 2 ** 20


def need_psutil():
    try:
        import psutil
        return psutil
    except ImportError:
        sys.exit("psutil 이 필요하다: pip install psutil")


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def summary(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0, "median": None, "p90": None, "max": None}
    return {"n": len(xs), "median": round(statistics.median(xs)), "p90": round(pct(xs, 0.9)), "max": round(max(xs))}


# ---------- 1) 부품별 메모리 ----------
PROBE = r'''
import json, sys, time, os, gc
sys.path[:0] = [r"{root}/src/pipeline", r"{root}/src/geo", r"{root}/src/vlm"]
import psutil
p = psutil.Process()
out = []
def mark(name, t0):
    gc.collect()
    out.append({{"step": name, "rss_mb": round(p.memory_info().rss / 2**20), "sec": round(time.perf_counter() - t0, 2)}})
t = time.perf_counter(); mark("파이썬 시작", t)
t = time.perf_counter(); import numpy, PIL; mark("numpy, PIL", t)
t = time.perf_counter(); import torch; torch.set_num_threads({cores}); import ultralytics; mark("torch, ultralytics 불러오기", t)
if {weights!r}:
    t = time.perf_counter(); from detect import Detector; d = Detector({weights!r}); mark("탐지 모델 올리고 한 번 돌림", t)
if {landuse!r}:
    t = time.perf_counter(); from query import GeoData; g = GeoData({landuse!r}, {buildings!r} or None, []); mark("용도지역·건물", t)
if {terrain!r}:
    t = time.perf_counter(); from sim_terrain import SimTerrain; s = SimTerrain({terrain!r}); mark("지형 격자", t)
print("PROBE" + json.dumps(out, ensure_ascii=False))
'''


def probe(args, run_args, env, pin):
    def val(flag):
        if flag in run_args:
            i = run_args.index(flag)
            return run_args[i + 1] if i + 1 < len(run_args) else None
        return None
    geo = val("--geo-dir")
    landuse = val("--landuse") or (str(Path(geo) / "landuse_5179.geojson") if geo else None)
    buildings = val("--buildings") or (str(Path(geo) / "buildings_5179.geojson") if geo else None)
    if buildings and not Path(buildings).is_file():
        buildings = None
    code = PROBE.format(root=str(ROOT).replace("\\", "/"), cores=args.cores, weights=val("--weights"),
                        landuse=landuse, buildings=buildings, terrain=val("--sim-terrain"))
    r = subprocess.run(pin + [sys.executable, "-c", code], capture_output=True, text=True, env=env, encoding="utf-8")
    line = next((l for l in r.stdout.splitlines() if l.startswith("PROBE")), None)
    if not line:
        print(r.stdout[-2000:], r.stderr[-2000:])
        return []
    rows = json.loads(line[5:])
    prev = None
    for row in rows:
        row["delta_mb"] = None if prev is None else row["rss_mb"] - prev
        prev = row["rss_mb"]
    return rows


# ---------- 2) 파이프라인 전체 ----------
class Sampler(threading.Thread):
    """프로세스(자식 포함)와 llama-server 의 메모리를 dt 초마다 잰다."""

    def __init__(self, psutil, pid, dt=0.1):
        super().__init__(daemon=True)
        self.ps, self.pid, self.dt = psutil, pid, dt
        self.peak = self.llama_peak = 0
        self.samples = []
        self.stop = False
        self.cpu = []

    def _tree_rss(self):
        try:
            root = self.ps.Process(self.pid)
            procs = [root] + root.children(recursive=True)
        except self.ps.NoSuchProcess:
            return None
        total = 0
        for p in procs:
            try:
                total += p.memory_info().rss
            except self.ps.Error:
                pass
        return total

    def _llama_rss(self):
        best = 0
        for p in self.ps.process_iter(["name", "memory_info"]):
            if "llama-server" in (p.info["name"] or ""):
                mi = p.info["memory_info"]
                best = max(best, getattr(mi, "rss", 0))
        return best

    def run(self):
        t0 = time.perf_counter()
        n = 0
        while not self.stop:
            rss = self._tree_rss()
            if rss is None:
                break
            self.peak = max(self.peak, rss)
            if n % 10 == 0:              # llama-server 찾기는 느려서 1초에 한 번
                self.llama_peak = max(self.llama_peak, self._llama_rss())
                self.samples.append((round(time.perf_counter() - t0, 1), round(rss / MB)))
            n += 1
            time.sleep(self.dt)


def run_pipeline(args, run_args, env, pin, psutil):
    cmd = pin + [sys.executable, str(ROOT / "src" / "pipeline" / "run.py")] + run_args
    print("[측정] " + " ".join(cmd))
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                            text=True, encoding="utf-8", errors="replace")
    if platform.system() == "Windows":
        try:
            psutil.Process(proc.pid).cpu_affinity(list(range(args.cores)))
        except Exception as e:
            print(f"[주의] 코어를 묶지 못함: {e}")
    s = Sampler(psutil, proc.pid)
    s.start()
    log = []
    for line in proc.stdout:
        log.append(line)
        if args.verbose or any(k in line for k in ("[요약]", "[형식 검사]", "통과", "판정:", "Traceback", "Error")):
            print("  | " + line.rstrip())
    proc.wait()
    s.stop = True
    s.join()
    return {"returncode": proc.returncode, "wall_sec": round(time.perf_counter() - t0, 1),
            "peak_rss_mb": round(s.peak / MB), "llama_peak_mb": round(s.llama_peak / MB) if s.llama_peak else None,
            "rss_timeline": s.samples, "log_tail": "".join(log[-40:])}


def collect(out):
    """알람 기록(alerts.jsonl), 탐지 결과, debug.jsonl 에서 단계별 시간을 모은다."""
    out = Path(out)
    det = []
    for f in sorted((out / "detections").glob("*.json")):
        try:
            det.append((json.load(open(f, encoding="utf-8")).get("latency_ms") or {}).get("detect"))
        except (OSError, ValueError):
            pass
    first, last = {}, {}
    for line in open(out / "alerts.jsonl", encoding="utf-8"):
        try:
            a = json.loads(line)
        except ValueError:
            continue
        lat = a.get("latency_ms") or {}
        key = (a["alert_id"], a.get("detection_id"))
        first.setdefault(key, lat)
        if lat.get("alert_to_description") is not None:
            last[key] = lat
    pub = list(first.values())
    desc = list(last.values())
    p_ms = []
    for line in open(out / "debug.jsonl", encoding="utf-8"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("p_smoke_ms") is not None:
            p_ms.append(d["p_smoke_ms"])
    total = [l.get("detect_to_alert", 0) + l["alert_to_description"] for l in desc
             if l.get("detect_to_alert") is not None]
    return {
        "frames": len(det),
        "알람 발행(갱신 포함)": len(pub),
        "stages_ms": {
            "탐지 (장마다)": summary(det),
            "좌표 변환": summary([l.get("geo_transform") for l in pub]),
            "지리 조회": summary([l.get("geo_query") for l in pub]),
            "판정": summary([l.get("verdict") for l in pub]),
            "탐지 → 알람": summary([l.get("detect_to_alert") for l in pub]),
            "검증기 (탐지마다)": summary(p_ms),
            "VLM 설명": summary([l.get("vlm") for l in desc]),
            "알람 → 설명": summary([l.get("alert_to_description") for l in desc]),
            "탐지 → 설명 (합계)": summary(total),
        },
    }


def fmt_ms(v):
    if v is None:
        return "-"
    return f"{v / 1000:.1f}초" if v >= 1000 else f"{v}ms"


def report(path, args, env_info, comp, run, stats):
    L = ["# 현장 기기 동작 조건 측정 결과 (7.3)", "",
         f"측정: {time.strftime('%Y-%m-%d %H:%M')} / `tools/measure_device.py`", "",
         "## 측정 환경", "", "| 항목 | 값 |", "|---|---|"]
    for k, v in env_info.items():
        L.append(f"| {k} | {v} |")
    L += ["", "## 부품별 메모리", "", "| 단계 | 누적 RSS | 늘어난 양 | 걸린 시간 |", "|---|---|---|---|"]
    for r in comp:
        L.append(f"| {r['step']} | {r['rss_mb']}MB | {'' if r['delta_mb'] is None else str(r['delta_mb']) + 'MB'} | {r['sec']}초 |")
    budget = args.mem_gb * 1024
    tot = run["peak_rss_mb"] + (run["llama_peak_mb"] or 0)
    L += ["", "## 파이프라인 전체", "",
          "| 항목 | 값 |", "|---|---|",
          f"| 프레임 | {stats['frames']}장 |",
          f"| 알람 발행 (갱신 포함) | {stats['알람 발행(갱신 포함)']}번 |",
          f"| 전체 걸린 시간 | {run['wall_sec']}초 |",
          f"| 파이프라인 최대 메모리 | {run['peak_rss_mb']}MB |",
          f"| llama-server 최대 메모리 | {str(run['llama_peak_mb']) + 'MB' if run['llama_peak_mb'] else '측정 안 함 (llama-server 없음)'} |",
          f"| 합계 / 기준 | {tot}MB / {budget}MB ({tot / budget * 100:.0f}%) |", "",
          "| 단계 | 중앙값 | 90% | 최대 | 개수 |", "|---|---|---|---|---|"]
    for k, s in stats["stages_ms"].items():
        L.append(f"| {k} | {fmt_ms(s['median'])} | {fmt_ms(s['p90'])} | {fmt_ms(s['max'])} | {s['n']} |")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump({"env": env_info, "components": comp, "run": run, "stats": stats},
              open(Path(path).with_suffix(".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description="현장 기기 동작 조건 측정 (7.3)")
    ap.add_argument("--cores", type=int, default=4, help="묶을 코어 수 (기준 4)")
    ap.add_argument("--mem-gb", type=float, default=8, help="견줄 메모리 기준 (기준 8)")
    ap.add_argument("--report", default="results/device_measure.md")
    ap.add_argument("--skip-probe", action="store_true", help="부품별 메모리를 건너뛴다")
    ap.add_argument("--verbose", action="store_true", help="run.py 출력을 모두 보인다")
    ap.add_argument("run_args", nargs=argparse.REMAINDER, help="-- 뒤에 run.py 인자")
    a = ap.parse_args()
    run_args = a.run_args[1:] if a.run_args[:1] == ["--"] else a.run_args
    if "--out" not in run_args:
        run_args += ["--out", "results/device_run"]
    out = run_args[run_args.index("--out") + 1]
    psutil = need_psutil()

    have = os.cpu_count() or 1
    cores = min(a.cores, have)
    if cores < a.cores:
        print(f"[주의] 이 기기는 {have}코어라 {a.cores}코어 기준을 채우지 못한다. {cores}코어로 잰다 (기준보다 불리)")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = str(cores)
    env["CUDA_VISIBLE_DEVICES"] = ""          # GPU 사용 안 함
    import shutil
    pin = ["taskset", "-c", f"0-{cores - 1}"] if platform.system() == "Linux" and shutil.which("taskset") else []
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    cpu_name = platform.processor() or ""
    if platform.system() == "Linux":
        try:
            cpu_name = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
        except (OSError, StopIteration):
            pass
    env_info = {"OS": f"{platform.system()} {platform.release()}", "CPU": cpu_name,
                "코어 (기기 / 묶은 수)": f"{have} / {cores}", "메모리 (기기 전체)": f"{vm.total / GB:.1f}GB",
                "스왑": f"{sw.total / GB:.1f}GB", "GPU": "사용 안 함 (CUDA_VISIBLE_DEVICES 비움)",
                "기준": f"{a.cores}코어, {a.mem_gb:g}GB, 스왑 없음, GPU 없음",
                "run.py 인자": " ".join(run_args)}

    comp = [] if a.skip_probe else probe(a, run_args, env, pin)
    run = run_pipeline(a, run_args, env, pin, psutil)
    if run["returncode"] != 0:
        print(run["log_tail"])
        sys.exit(f"run.py 가 실패했다 (코드 {run['returncode']})")
    stats = collect(out)
    report(a.report, a, env_info, comp, run, stats)


if __name__ == "__main__":
    main()
