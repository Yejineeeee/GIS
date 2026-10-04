"""파이프라인 점검용 입력 생성기.

이미지 폴더를 파일 이름 순서로 읽어, 프레임마다 탐지 결과 JSON(docs/schema.md v0.2)을
1~2초 간격으로 내보낸다. 탐지 값은 가짜지만 형식은 실제 탐지 모델 출력과 같다.
탐지 모델이 없어도 지리 조회, 알람, 대시보드를 만들고 시험할 수 있게 하는 용도다.

사용 예:
    # 가짜 이미지 10장 만들고 바로 재생
    python tools/replay.py --images results/fake_frames --make-fake 10 --out results/replay

    # 실제 이미지 폴더 재생, 1초 간격
    python tools/replay.py --images data/frames/V0012 --interval 1 --out results/replay

    # 기다리지 않고 바로 전부 출력 (테스트용)
    python tools/replay.py --images results/fake_frames --no-sleep

출력:
    --out 폴더에 000000.json, 000001.json ... 을 순서대로 저장하고,
    같은 내용을 한 줄씩 표준 출력(JSON Lines)으로도 내보낸다.
"""
import argparse
import json
import math
import random
import struct
import sys
import time
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

ap = argparse.ArgumentParser(description="파이프라인 점검용 입력 생성기")
ap.add_argument("--images", required=True, help="이미지 폴더")
ap.add_argument("--out", default=None, help="JSON 저장 폴더 (없으면 표준 출력만)")
ap.add_argument("--interval", type=float, default=1.5, help="프레임 간격(초), 기본 1.5")
ap.add_argument("--no-sleep", action="store_true", help="간격 없이 바로 출력")
ap.add_argument("--video-id", default="V0000", help="영상 번호")
ap.add_argument("--lat", type=float, default=37.5121, help="드론 시작 위도")
ap.add_argument("--lon", type=float, default=127.0412, help="드론 시작 경도")
ap.add_argument("--alt", type=float, default=100.0, help="고도(m, 지면 기준)")
ap.add_argument("--heading", type=float, default=90.0, help="진행 방향(도, 진북 0)")
ap.add_argument("--speed", type=float, default=5.0, help="드론 속도(m/s)")
ap.add_argument("--pitch", type=float, default=-45.0, help="촬영 각도(도, -90 수직 하향)")
ap.add_argument("--hfov", type=float, default=84.0, help="수평 화각(도)")
ap.add_argument("--empty-prob", type=float, default=0.0, help="탐지가 없는 프레임 비율(0~1)")
ap.add_argument("--seed", type=int, default=0, help="난수 시드 (같은 값이면 같은 결과)")
ap.add_argument("--make-fake", type=int, default=0, help="이미지 폴더에 가짜 이미지 N장 생성")
args = ap.parse_args()

rng = random.Random(args.seed)
img_dir = Path(args.images)
exts = {".jpg", ".jpeg", ".png"}

# 1) 가짜 이미지 만들기 (Pillow 없이 표준 라이브러리로 PNG 작성)
if args.make_fake > 0:
    img_dir.mkdir(parents=True, exist_ok=True)
    w, h = 1280, 720
    for i in range(args.make_fake):
        shade = 60 + i * 15 % 150
        row = b"\x00" + bytes([shade, shade + 30 if shade < 225 else 255, shade]) * w
        raw = row * h
        png = b"\x89PNG\r\n\x1a\n"
        for tag, data in (
            (b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)),
            (b"IDAT", zlib.compress(raw, 9)),
            (b"IEND", b""),
        ):
            png += struct.pack(">I", len(data)) + tag + data
            png += struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        (img_dir / f"fake_{i:04d}.png").write_bytes(png)
    print(f"[replay] 가짜 이미지 {args.make_fake}장 생성: {img_dir}", file=sys.stderr)

# 2) 이미지 목록 (파일 이름 순서 = 프레임 순서)
if not img_dir.is_dir():
    sys.exit(f"[replay] 이미지 폴더가 없다: {img_dir}")
files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts)
if not files:
    sys.exit(f"[replay] 이미지가 없다: {img_dir}")

out_dir = Path(args.out) if args.out else None
if out_dir:
    out_dir.mkdir(parents=True, exist_ok=True)

kst = timezone(timedelta(hours=9))
t0 = datetime.now(kst).replace(microsecond=0)
m_per_deg_lat = 111_320.0
m_per_deg_lon = 111_320.0 * math.cos(math.radians(args.lat))

print(f"[replay] {len(files)}프레임, 간격 {args.interval}s", file=sys.stderr)

for frame_id, path in enumerate(files):
    t_start = time.perf_counter()

    # 3) 이미지 크기 읽기 (PNG는 헤더, JPEG는 SOF 마커, 실패하면 1920x1080)
    width, height = 1920, 1080
    head = path.read_bytes()
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        width, height = struct.unpack(">II", head[16:24])
    elif head[:2] == b"\xff\xd8":
        i = 2
        while i + 9 < len(head):
            if head[i] != 0xFF:
                break
            marker = head[i + 1]
            seg_len = struct.unpack(">H", head[i + 2:i + 4])[0]
            if marker in (0xC0, 0xC1, 0xC2):
                height, width = struct.unpack(">HH", head[i + 5:i + 9])
                break
            i += 2 + seg_len

    # 4) 드론 위치: 시작점에서 진행 방향으로 등속 이동
    dist = args.speed * args.interval * frame_id
    lat = args.lat + dist * math.cos(math.radians(args.heading)) / m_per_deg_lat
    lon = args.lon + dist * math.sin(math.radians(args.heading)) / m_per_deg_lon

    # 5) 가짜 탐지: 박스 안에 들어가는 다각형, 넓이는 신발끈 공식
    detections = []
    if rng.random() >= args.empty_prob:
        bw = rng.randint(width // 10, width // 4)
        bh = rng.randint(height // 10, height // 4)
        x = rng.randint(0, width - bw)
        y = rng.randint(0, height - bh)
        poly = [
            [x + rng.randint(0, bw // 4), y],
            [x + bw, y + rng.randint(0, bh // 4)],
            [x + bw - rng.randint(0, bw // 4), y + bh],
            [x, y + bh - rng.randint(0, bh // 4)],
        ]
        area = 0.0
        for k in range(len(poly)):
            x1, y1 = poly[k]
            x2, y2 = poly[(k + 1) % len(poly)]
            area += x1 * y2 - x2 * y1
        area_px = int(abs(area) / 2)
        detections.append({
            "id": f"{args.video_id}-{frame_id}-0",
            "class": "smoke",
            "score": round(rng.uniform(0.5, 0.99), 2),
            "bbox": [x, y, bw, bh],
            "polygon": poly,
            "area_px": area_px,
            "area_ratio": round(area_px / (width * height), 4),
        })

    msg = {
        "schema_version": "0.2",
        "video_id": args.video_id,
        "frame_id": frame_id,
        "time": (t0 + timedelta(seconds=args.interval * frame_id)).isoformat(timespec="milliseconds"),
        "image": path.as_posix(),
        "image_size": {"width": width, "height": height},
        "detections": detections,
        "camera": {
            "lat": round(lat, 7),
            "lon": round(lon, 7),
            "alt_m": args.alt,
            "alt_ref": "AGL",
            "heading_deg": args.heading,
            "pitch_deg": args.pitch,
            "hfov_deg": args.hfov,
            "source": "virtual",
        },
        "latency_ms": {"detect": round((time.perf_counter() - t_start) * 1000)},
    }

    # 6) 내보내기: 파일 + 표준 출력 한 줄
    if out_dir:
        (out_dir / f"{frame_id:06d}.json").write_text(
            json.dumps(msg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(msg, ensure_ascii=False), flush=True)

    if not args.no_sleep and frame_id < len(files) - 1:
        time.sleep(args.interval)

print(f"[replay] 완료: {len(files)}개", file=sys.stderr)
