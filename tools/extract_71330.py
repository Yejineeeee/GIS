"""선별 목록(select_71330.csv)에 있는 300장만 VS.zip, VL.zip에서 꺼낸다.

압축을 전부 풀지 않고 목록에 있는 파일만 꺼내므로 저장 공간은 zip 크기 + 약 1GB면 된다.

사용법:
    python3 tools/extract_71330.py docs/select_71330.csv <VS.zip 경로> <VL.zip 경로> <출력 폴더>

출력:
    <출력 폴더>/images/train/*.jpg, images/val/*.jpg   (원천 사진)
    <출력 폴더>/labels/train/*.json, labels/val/*.json (라벨)
    <출력 폴더>/select_71330.csv                         (목록 복사본)
"""
import csv
import shutil
import sys
import unicodedata
import zipfile
from pathlib import Path


def nfc(s):
    # 맥은 한글 파일 이름을 자모로 쪼개 저장하기도 해서(NFD) 비교 전에 한 형태로 맞춘다
    return unicodedata.normalize("NFC", s)


def real_name(info):
    # 한글 이름이 cp949로 저장된 zip은 파이썬이 cp437로 잘못 읽으므로 되돌린다
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode("cp437").decode("cp949")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename


def extract(zip_path, wanted, out_root, kind):
    """wanted: {파일 이름: split}. 찾은 파일 이름 집합을 돌려준다."""
    found = set()
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            base = nfc(real_name(info).split("/")[-1])
            if base not in wanted or base in found:
                continue
            dest = out_root / kind / wanted[base] / base
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            found.add(base)
            print(f"\r{kind}: {len(found)}/{len(wanted)}", end="", flush=True)
    print()
    return found


def main():
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)
    csv_path, vs_zip, vl_zip, out = sys.argv[1:]
    out_root = Path(out)

    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    images = {nfc(r["image_file"]): r["split"] for r in rows}
    labels = {nfc(r["file_name"]): r["split"] for r in rows}

    got_img = extract(vs_zip, images, out_root, "images")
    got_lbl = extract(vl_zip, labels, out_root, "labels")

    out_root.mkdir(parents=True, exist_ok=True)
    shutil.copy(csv_path, out_root / "select_71330.csv")

    miss_img = sorted(set(images) - got_img)
    miss_lbl = sorted(set(labels) - got_lbl)
    print(f"사진 {len(got_img)}/{len(images)}, 라벨 {len(got_lbl)}/{len(labels)}")
    for name in miss_img[:10]:
        print("  사진 없음:", name)
    for name in miss_lbl[:10]:
        print("  라벨 없음:", name)
    if miss_img or miss_lbl:
        sys.exit(2)
    print("완료")


if __name__ == "__main__":
    main()
