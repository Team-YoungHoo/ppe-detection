import hashlib
import random
from pathlib import Path
from PIL import Image
import imagehash

# ----- 설정값 (기능 명세: train.seed = 42) -----
SEED = 42
random.seed(SEED)

ROOT = Path("data/processed/construction safety.v2-release.yolov8")
SPLITS = ["train", "valid", "test"]
IMG_EXT = ".jpg"
PHASH_THRESHOLD = 5  # 해밍 거리 <= 5 면 "비슷한 프레임"으로 간주 (경험적 기준)

def list_images(split):
    """증강으로 새로 만든 _aug/_bal 파일은 train에만 있는 합성본이라 제외.
    원본 프레임 간의 분할 누수만 확인한다."""
    img_dir = ROOT / split / "images"
    return [p for p in img_dir.glob(f"*{IMG_EXT}")
            if "_aug" not in p.stem and "_bal" not in p.stem]

def md5_of(path):
    return hashlib.md5(path.read_bytes()).hexdigest()

def phash_of(path):
    return imagehash.phash(Image.open(path))

# ---------- 1. 분할별 이미지 장수 ----------
print("=== 분할별 이미지 장수 ===")
split_images = {}
for split in SPLITS:
    imgs = list_images(split)
    split_images[split] = imgs
    print(f"  {split}: {len(imgs)}장 (증강분 제외, 원본 기준)")

# ---------- 2. 완전 동일 중복 검사 (MD5) ----------
print("\n=== 완전 동일 중복 검사 (파일 내용 기준) ===")
hash_to_files = {}
for split in SPLITS:
    for img in split_images[split]:
        h = md5_of(img)
        hash_to_files.setdefault(h, []).append((split, img))

exact_dupes = {h: v for h, v in hash_to_files.items() if len(v) > 1}
if exact_dupes:
    print(f"  완전 동일 중복 {len(exact_dupes)}건 발견:")
    for h, files in exact_dupes.items():
        for split, path in files:
            print(f"    [{split}] {path.name}")
else:
    print("  완전 동일 중복 없음")

# ---------- 3. 비슷한 프레임 검사 (perceptual hash, split 간 비교만) ----------
print("\n=== 비슷한 프레임(같은 영상 추정) 검사 ===")
phashes = {split: [(p, phash_of(p)) for p in split_images[split]] for split in SPLITS}

near_dupes = []  # (train_or_other_split, path, other_split, other_path, distance)
pairs = [("train", "valid"), ("train", "test"), ("valid", "test")]
for a, b in pairs:
    for pa, ha in phashes[a]:
        for pb, hb in phashes[b]:
            dist = ha - hb
            if dist <= PHASH_THRESHOLD:
                near_dupes.append((a, pa, b, pb, dist))

if near_dupes:
    print(f"  비슷한 프레임 {len(near_dupes)}쌍 발견 (해밍 거리 <= {PHASH_THRESHOLD}):")
    for a, pa, b, pb, dist in near_dupes:
        print(f"    [{a}] {pa.name}  <->  [{b}] {pb.name}  (거리={dist})")
else:
    print("  비슷한 프레임 없음")

# ---------- 예외처리: train 우선, valid/test 쪽 제거 ----------
to_remove = set()  # (split, path) — valid 또는 test 쪽만 담김

def mark_for_removal(split_a, path_a, split_b, path_b):
    # train은 항상 보존. train이 아닌 쪽, 혹은 둘 다 train이 아니면 valid를 우선 보존하고 test를 제거
    priority = {"train": 0, "valid": 1, "test": 2}
    if priority[split_a] < priority[split_b]:
        to_remove.add((split_b, path_b))
    else:
        to_remove.add((split_a, path_a))

for h, files in exact_dupes.items():
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            mark_for_removal(files[i][0], files[i][1], files[j][0], files[j][1])

for a, pa, b, pb, dist in near_dupes:
    mark_for_removal(a, pa, b, pb)

print(f"\n=== 제거 대상 (valid/test 중복, train 보존) ===")
if to_remove:
    for split, path in sorted(to_remove):
        print(f"  [{split}] {path.name} -> 제거 예정")

    confirm = input("\n위 파일들을 valid/test에서 실제로 삭제할까요? (y/n): ")
    if confirm.lower() == "y":
        for split, path in to_remove:
            label_path = ROOT / split / "labels" / f"{path.stem}.txt"
            path.unlink()
            if label_path.exists():
                label_path.unlink()
            print(f"  삭제됨: [{split}] {path.name}")
    else:
        print("  삭제를 건너뛰었습니다. 목록만 확인하세요.")
else:
    print("  제거할 중복 없음")

print("\n=== 최종 분할별 이미지 장수 (제거 후, 증강분 제외) ===")
for split in SPLITS:
    n = len(list_images(split))
    print(f"  {split}: {n}장")