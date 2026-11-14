import sys
from pathlib import Path

root = Path(sys.argv[1])
NO_HELMET_ID = 3

targets = []
for lb in (root / "labels").glob("*.txt"):
    lines = [l for l in lb.read_text().splitlines() if l.strip()]
    if any(int(l.split()[0]) == NO_HELMET_ID for l in lines):
        targets.append(lb.stem)

print(f"no-helmet 포함 이미지: {len(targets)}장")
Path("no_helmet_list.txt").write_text("\n".join(targets), encoding="utf-8")