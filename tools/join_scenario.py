# -*- coding: utf-8 -*-
"""신고 시나리오의 구간별 프레임(수색 비행, 접근 1, 접근 2 ...)을 한 폴더로 잇는다.

구간마다 sim_0000 부터 새로 번호가 붙으므로, 순서대로 이어 번호와 frame_id 를 다시
매긴다. 시간(time)은 각 구간이 찍을 때 적은 장면 시간을 그대로 쓴다. 영상에 쓸 자막
(caption)과 정답까지 거리(sweep.D)를 메타에 넣는다. 정답은 채점과 자막에만 쓰고
드론 비행에는 쓰지 않았다.

    python tools/join_scenario.py sim/frames_scenario sim/frames_search:0:2:SEARCH sim/frames_leg1 sim/frames_leg2
    (폴더:처음:끝:자막, 처음과 끝은 프레임 번호, 비우면 전부, 자막 기본은 APPROACH)
"""
import json
import math
import shutil
import sys
from pathlib import Path


def main():
    out = Path(sys.argv[1])
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    n = 0
    for spec in sys.argv[2:]:
        folder, a, b, cap = (spec.split(":") + ["", "", ""])[:4]
        pngs = sorted(Path(folder).glob("sim_*.png"))
        lo = int(a) if a else 0
        hi = int(b) if b else len(pngs) - 1
        for png in pngs[lo:hi + 1]:
            meta = json.load(open(png.with_suffix(".json"), encoding="utf-8"))
            c, t = meta["camera"], meta["truth"]
            D = math.hypot((t["fire_lat"] - c["lat"]) * 111320,
                           (t["fire_lon"] - c["lon"]) * 111320 * math.cos(math.radians(c["lat"])))
            meta["frame_id"] = n
            meta["video_id"] = "SCENARIO_VW3D"
            meta.setdefault("sweep", {})["D"] = round(D, 1)
            ts = (meta.get("search") or meta.get("approach") or {}).get("t_s", 0)
            meta["stage"] = cap or "APPROACH"          # 자막 단계 (tools/drone_overlay.py)
            meta["caption"] = f"{meta['stage']}  t={ts:.0f}s  fire {D:.0f} m"
            shutil.copy(png, out / f"sim_{n:04d}.png")
            json.dump(meta, open(out / f"sim_{n:04d}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            n += 1
    print(f"{n}장 -> {out}")


if __name__ == "__main__":
    main()
