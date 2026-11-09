from pathlib import Path
import random
import cv2

# 프로젝트 루트 폴더
BASE_DIR = Path(__file__).resolve().parents[2]

# 변환된 데이터셋 폴더
DATASET_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "construction safety.v2-release.yolov8"
)

# 데이터셋 구분
SPLITS = ["train", "valid", "test"]

def draw_boxes(image_path, label_path, output_path):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"이미지 읽기 실패: {image_path}")
        return

    height, width = image.shape[:2]

    class_names = [
        "person",
        "helmet",
        "vest",
        "no-helmet",
        "no-vest"
    ]

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255)
    ]

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            x_center, y_center, box_width, box_height = map(float, parts[1:])

            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height

            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            color = colors[class_id]

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            label = class_names[class_id]

            cv2.putText(
                image,
                label,
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

    cv2.imwrite(str(output_path), image)

# 이미지 및 라벨 폴더 확인
for split in SPLITS:
    images_dir = DATASET_DIR / split / "images"
    labels_dir = DATASET_DIR / split / "labels"

    print(f"\n[{split}]")

    print("이미지 폴더 존재:", images_dir.exists())
    print("라벨 폴더 존재:", labels_dir.exists())

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

    image_files = []

    for extension in image_extensions:
        image_files.extend(images_dir.glob(f"*{extension}"))

    label_files = list(labels_dir.glob("*.txt"))

    print("이미지 개수:", len(image_files))
    print("라벨 파일 개수:", len(label_files))

    # 무작위 이미지 20장 시각화
VISUAL_DIR = BASE_DIR / "data" / "visualized"
VISUAL_DIR.mkdir(parents=True, exist_ok=True)

all_samples = []

for split in SPLITS:
    images_dir = DATASET_DIR / split / "images"
    labels_dir = DATASET_DIR / split / "labels"

    for image_path in images_dir.iterdir():
        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue

        label_path = labels_dir / (image_path.stem + ".txt")

        if label_path.exists():
            all_samples.append((image_path, label_path))

sample_count = min(20, len(all_samples))
selected_samples = random.sample(all_samples, sample_count)

for index, (image_path, label_path) in enumerate(selected_samples, start=1):
    output_path = VISUAL_DIR / f"sample_{index:02d}.jpg"

    draw_boxes(image_path, label_path, output_path)

print(f"\n시각화 완료: {sample_count}장")
print(f"저장 위치: {VISUAL_DIR}")