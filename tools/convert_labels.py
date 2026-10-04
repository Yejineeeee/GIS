# -*- coding: utf-8 -*-
"""AI Hub 71330 라벨을 분할 모델 학습용으로 바꾼다.

실제 라벨 파일을 아직 보지 못한 상태로 작성했다. 그래서 먼저 구조를 확인하는
모드를 두었고, 가정이 틀리면 조용히 넘어가지 않고 멈추게 했다.

    python convert_labels.py --inspect "라벨 폴더"      구조와 클래스 이름만 본다
    python convert_labels.py "라벨 폴더" "출력 폴더"     변환한다

출력은 YOLO 분할 형식이다. 이미지 한 장당 txt 한 개, 한 줄이 객체 하나다.
    클래스번호 x1 y1 x2 y2 ...   (모두 0에서 1 사이로 정규화)
"""
import sys, os, json, random, shutil
from collections import Counter

# 학습에 쓸 클래스. 연기 한 종류로 합친다. 색을 나누지 않는 것은 제안서와
# docs/schema.md v0.2 가 클래스를 smoke 하나로 두기 때문이다.
# 화염은 제외한다. 근거를 남기려고 건너뛴 수만 센다.
KEEP = {
    "흑색연기": 0,
    "백색및회색연기": 0,
    "백색 및 회색연기": 0,
    "백색과 회색연기": 0,
}
VAL_RATIO = 0.2
SEED = 42

args = [a for a in sys.argv[1:] if not a.startswith("--")]
INSPECT = "--inspect" in sys.argv
ROOT = args[0] if args else None
OUT = args[1] if len(args) > 1 else "data/yolo"

if not ROOT:
    raise SystemExit(__doc__)

files = [os.path.join(dp, n) for dp, _, ns in os.walk(ROOT)
         for n in ns if n.lower().endswith(".json")]
if not files:
    raise SystemExit(f"{ROOT} 에서 json을 찾지 못했습니다.")
print(f"라벨 파일 {len(files)}개")


def load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  읽기 실패 {os.path.basename(path)}: {e}")
        return None


def env_of(label):
    env = label.get("environment", {})
    if isinstance(env, list):
        env = env[0] if env else {}
    return env if isinstance(env, dict) else {}


def is_drone(label):
    d = str(env_of(label).get("device", ""))
    return "드론" in d or "drone" in d.lower()


# ── 구조 확인 모드
if INSPECT:
    sample = load(files[0])
    print(f"\n최상위 키: {list(sample.keys())}")
    for key in ("images", "annotations", "categories"):
        v = sample.get(key)
        if isinstance(v, list) and v:
            print(f"\n{key}[0] 키: {list(v[0].keys())}")
            print(f"  {json.dumps(v[0], ensure_ascii=False)[:400]}")
        elif v is not None:
            print(f"\n{key}: {json.dumps(v, ensure_ascii=False)[:200]}")
    print(f"\nenvironment: {json.dumps(env_of(sample), ensure_ascii=False)[:400]}")

    names, seg_keys, drone = Counter(), Counter(), 0
    for path in files:
        label = load(path)
        if not label:
            continue
        if is_drone(label):
            drone += 1
        for c in label.get("categories", []):
            names[c.get("name")] += 1
        for a in label.get("annotations", [])[:1]:
            seg_keys[tuple(sorted(a.keys()))] += 1
    print(f"\n클래스 이름: {dict(names)}")
    print(f"드론 촬영분: {drone} / {len(files)}")
    print("\nannotation 키 조합:")
    for k, n in seg_keys.most_common():
        print(f"  {n}건  {k}")
    print("\n위 클래스 이름이 스크립트 위쪽 KEEP 과 다르면 KEEP 을 고치세요.")
    raise SystemExit(0)


# ── 변환
def polygon_of(ann):
    """폴리곤 좌표를 [(x, y), ...] 로 꺼낸다. 없으면 bbox로 사각형을 만든다."""
    seg = ann.get("segmentation") or ann.get("polygon")
    if seg:
        flat = seg[0] if (isinstance(seg, list) and seg and isinstance(seg[0], list)) else seg
        if flat and isinstance(flat[0], (list, tuple)):        # [[x, y], ...]
            return [(float(p[0]), float(p[1])) for p in flat]
        return [(float(flat[i]), float(flat[i+1])) for i in range(0, len(flat) - 1, 2)]
    box = ann.get("bbox")
    if box and len(box) == 4:
        x, y, w, h = map(float, box)
        return [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
    return None


random.seed(SEED)
for split in ("train", "val"):
    os.makedirs(os.path.join(OUT, "labels", split), exist_ok=True)
    os.makedirs(os.path.join(OUT, "images", split), exist_ok=True)

stat = Counter()
unknown = Counter()
pairs = []

for path in files:
    label = load(path)
    if not label:
        stat["읽기실패"] += 1
        continue
    if not is_drone(label):
        stat["드론아님"] += 1
        continue

    imgs = label.get("images") or []
    if not imgs:
        stat["이미지정보없음"] += 1
        continue
    img = imgs[0]
    W, H = img.get("width"), img.get("height")
    if not W or not H:
        stat["크기없음"] += 1
        continue

    names = {c.get("id"): c.get("name") for c in label.get("categories", [])}
    lines = []
    for ann in label.get("annotations", []):
        name = str(names.get(ann.get("category_id"), "")).strip()
        if name not in KEEP:
            unknown[name] += 1
            continue
        poly = polygon_of(ann)
        if not poly or len(poly) < 3:
            stat["폴리곤없음"] += 1
            continue
        coords = []
        for x, y in poly:
            coords += [f"{min(max(x/W, 0), 1):.6f}", f"{min(max(y/H, 0), 1):.6f}"]
        lines.append(f"{KEEP[name]} " + " ".join(coords))

    if not lines:
        stat["연기없음"] += 1
        continue
    pairs.append((img.get("file_name", ""), lines))
    stat["변환"] += 1

random.shuffle(pairs)
n_val = int(len(pairs) * VAL_RATIO)
if len(pairs) >= 5 and n_val == 0:
    n_val = 1          # 검증이 0장이 되는 것을 막는다
for i, (file_name, lines) in enumerate(pairs):
    split = "val" if i < n_val else "train"
    stem = os.path.splitext(os.path.basename(file_name))[0] or f"img_{i:05d}"
    with open(os.path.join(OUT, "labels", split, stem + ".txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

with open(os.path.join(OUT, "dataset.yaml"), "w", encoding="utf-8") as f:
    f.write(f"path: {os.path.abspath(OUT)}\ntrain: images/train\nval: images/val\n\n"
            "names:\n  0: smoke\n")

print("\n결과")
for k, v in stat.most_common():
    print(f"  {k}: {v}")
print(f"  학습 {len(pairs) - n_val}장, 검증 {n_val}장")
if unknown:
    print(f"\n건너뛴 클래스: {dict(unknown)}")
    print("  연기 클래스가 여기 있으면 KEEP 의 이름을 고치세요.")
print(f"\n라벨은 {OUT}/labels 아래에 있습니다.")
print("이미지는 라벨의 file_name 을 보고 같은 이름으로 images/train, images/val 에 넣으세요.")
