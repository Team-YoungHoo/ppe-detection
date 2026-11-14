import sys, collections
from pathlib import Path
import yaml

root = Path(sys.argv[1])
names = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))["names"]
if isinstance(names, dict):
    names = [names[k] for k in sorted(names)]

for split in ["train", "valid", "test"]:
    img_dir = root / split / "images"
    lbl_dir = root / split / "labels"
    imgs = [p for p in img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    cnt = collections.Counter()
    empty = 0
    for lb in lbl_dir.glob("*.txt"):
        lines = [l for l in lb.read_text().splitlines() if l.strip()]
        empty += (len(lines) == 0)
        for l in lines:
            cnt[int(l.split()[0])] += 1
    print(f"[{split}] 이미지 {len(imgs)}장, 빈 라벨 {empty}장")
    for i, n in enumerate(names):
        print(f"   {i} {n}: {cnt[i]}")