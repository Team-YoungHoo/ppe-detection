# train/scripts/dataset_stats_ppe.py
import sys, collections
from pathlib import Path
import yaml

root = Path(sys.argv[1])
yaml_path = next(root.glob("*.yaml"), None)
names = None
if yaml_path:
    names = yaml.safe_load(yaml_path.read_text(encoding="utf-8")).get("names")
if isinstance(names, dict):
    names = [names[k] for k in sorted(names)]
if not names:  # yaml이 없을 때 문서 기준 이름 사용
    names = ["helmet","gloves","vest","boots","goggles","none",
             "Person","no_helmet","no_goggle","no_gloves","no_boots"]

def label_dir(split):
    for p in (root/"labels"/split, root/split/"labels"):
        if p.exists(): return p
def image_dir(split):
    for p in (root/"images"/split, root/split/"images"):
        if p.exists(): return p

total = collections.Counter()
for split in ["train", "val", "tests"]:
    ld, idr = label_dir(split), image_dir(split)
    if not ld or not idr:
        print(f"[{split}] 폴더를 찾지 못함"); continue
    imgs = [p for p in idr.glob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}]
    cnt, empty = collections.Counter(), 0
    for lb in ld.glob("*.txt"):
        lines = [l for l in lb.read_text().splitlines() if l.strip()]
        empty += (len(lines) == 0)
        for l in lines:
            cnt[int(l.split()[0])] += 1
    total += cnt
    print(f"[{split}] 이미지 {len(imgs)}장, 빈 라벨 {empty}장, 박스 {sum(cnt.values())}개")
print("\n클래스별 박스 수 (전체):")
for i, n in enumerate(names):
    print(f"  {i:2d} {n:10s}: {total[i]}")