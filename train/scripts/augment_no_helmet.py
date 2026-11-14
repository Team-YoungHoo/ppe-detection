import cv2, albumentations as A
from pathlib import Path

SRC = Path("data/processed/construction safety.v2-release.yolov8/train")
IMG_EXT = ".jpg"   # 1번에서 확인한 확장자로 수정
N_COPIES = 4

transform = A.Compose([
    A.RandomBrightnessContrast(p=0.8),
    A.HueSaturationValue(p=0.5),
    A.Rotate(limit=8, p=0.5, border_mode=cv2.BORDER_REPLICATE),
    A.MotionBlur(blur_limit=3, p=0.2),
    A.RandomShadow(p=0.3),
], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

names = Path("no_helmet_list.txt").read_text(encoding="utf-8").splitlines()

for stem in names:
    img_path = SRC / "images" / f"{stem}{IMG_EXT}"
    lbl_path = SRC / "labels" / f"{stem}.txt"
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"이미지를 못 찾음: {img_path}")
        continue
    lines = [l.split() for l in lbl_path.read_text().splitlines() if l.strip()]
    bboxes = [[float(x) for x in l[1:]] for l in lines]
    labels = [int(l[0]) for l in lines]

    for c in range(N_COPIES):
        out = transform(image=img, bboxes=bboxes, class_labels=labels)
        new_stem = f"{stem}_aug{c}"
        cv2.imwrite(str(SRC / "images" / f"{new_stem}{IMG_EXT}"), out["image"])
        with open(SRC / "labels" / f"{new_stem}.txt", "w") as f:
            for cls, box in zip(out["class_labels"], out["bboxes"]):
                f.write(f"{int(cls)} {' '.join(f'{v:.6f}' for v in box)}\n")
print(f"{len(names)}장 x {N_COPIES}개 = {len(names)*N_COPIES}장 추가 생성")